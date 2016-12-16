import csv
import random
from os import path as osp
from random import shuffle

import gc
import numpy as np
from PIL import Image
from scipy.misc import imresize

from sign_detection.model.IdentifiedImage import IdentifiedImage
from sign_detection.model.RegionOfInterest import RegionOfInterest
from sign_detection.model.ScalingSlidingWindow import ScalingSlidingWindow


class BatchLoader(object):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    :ivar _image: IdentifiedImage
    """

    def __init__(self, params, num, fraction):
        """
        Initialize the batch loader. This already reads the list of images but generates no windows.
        :param params: The parameter dictionary containing the values batch_size, gtsdb_root and window_size
        :param num: The number of windows to go through before filtering some out. Note that the higher this number is,
                    the more random the results are but also more RAM will be used.
        :param fraction: The percentage of images with signs (eg. fraction == 0.3: signs: 100, no signs: 233)
        """
        self.gtsdb_root = params['gtsdb_root']
        self.window_size = params['window_size']
        self._cur = 0
        self._sliding_window = None
        self._image = None
        self.num = num
        self.fraction = fraction

        self.images = get_images_and_regions(self.gtsdb_root)

        print "BatchLoader initialized with {} images, {}% of each collection are signs".format(
            len(self.images), fraction*100)

    def next_window(self):
        """
        Get the next window. The windows are generated before, so it can take a while until this function returns
        :return: A new window
        :returns: list[list[list[float]]], int
        """
        try:
            return self._windows.pop()
        except:
            self.__collect_windows()

            # Make sure that all the unused images are removed from RAM
            gc.collect()

            # In case that no windows were generated, retry everything
            return self.next_window()

    def __collect_windows(self):
        """
        Generate a list of windows from windows
        :return: None
        :returns: None
        """
        signs = []
        no_signs = []

        for i in range(0, self.num, 1):
            image, label = self.__load_next_window()
            if label == 1:
                signs.append((image, label))
            else:
                no_signs.append((image, label))

        no_sign_count = int(max(0, min(len(signs) / self.fraction - len(signs), len(no_signs))))
        random.shuffle(no_signs)

        # create and shuffle result array
        result = []
        result.extend(signs)
        result.extend(no_signs[0:int(no_sign_count)])
        random.shuffle(result)
        self._windows = result

        # some debug info
        print "Collected {} images. Stored {} images, {} of these contain signs, {} of these contain no signs. " \
              "Overall {} images contain no signs.".format(
               self.num, len(result), len(signs), no_sign_count, len(no_signs))

    def generate_regions_from_image(self, size=64, regions_per_roi=10, regions_per_image=3, overlap=0.75):
        image, raw = self.__load_next_image()

        # generate regions from rois
        signs = []
        for roi in image.region_of_interests:
            dx = lambda dx: (roi.x1 - roi.x2) * (1 - dx) / 2 * (random.random() * 2 - 1)
            dy = lambda dy: (roi.y1 - roi.y2) * (1 - dy) / 2 * (random.random() * 2 - 1)

            generated = []
            i = 0
            while len(generated) < regions_per_roi and i < regions_per_roi * 2:
                i += 1
                d = dx(overlap)
                x1 = max(0, int(roi.x1 + d))
                x2 = max(0, int(roi.x2 + d))
                if x2 < x1:
                    temp = x1
                    x1 = x2
                    x2 = temp
                d = dy(overlap)
                y1 = max(0, int(roi.y1 + d))
                y2 = max(0, int(roi.y2 + d))
                if y2 < y1:
                    temp = y1
                    y1 = y2
                    y2 = temp

                region = RegionOfInterest(x1, y1, x2, y2, roi.sign)
                o = region.get_overlap(roi)
                if o >= overlap:
                    image_raw = raw[y1:y2 + 1, x1:x2 + 1, :]
                    scaled = imresize(image_raw, (size, size))
                    transposed = scaled.transpose((2, 0, 1))
                    generated.append((transposed, roi.sign))
            signs.extend(generated)

        # generate an equal amount of non sign images
        no_signs = self.generate_windows_without_signs(size=size, regions_per_image=regions_per_image, max_overlap=overlap/2, image=image, raw=raw)

        result = no_signs[:]
        result.extend(signs)
        shuffle(result)
        return result

    def generate_windows_without_signs(self, size=64, regions_per_image=3, max_overlap=0.1, transpose=(2, 0, 1),
                                       image=None, raw=None):
        if image == None or raw == None:
            image, raw = self.__load_next_image()
        no_signs = []
        while len(no_signs) < regions_per_image:
            local_size = random.randint(1, 400)
            x1 = random.randint(1, raw.shape[1] - local_size)
            y1 = random.randint(1, raw.shape[0] - local_size)

            flag = True
            region = RegionOfInterest(x1, y1, x1 + local_size, y1 + local_size, 0)
            for roi in image.region_of_interests:
                if region.get_overlap(roi) > max_overlap:
                    flag = False
            if flag:
                image_raw = raw[y1:y1 + local_size, x1:x1 + local_size, :]
                scaled = imresize(image_raw, (size, size))
                if transpose is not None:
                    transposed = scaled.transpose((2, 0, 1))
                    no_signs.append((transposed, 43))  # 43 Is the class for "no sign"
                else:
                    no_signs.append((scaled, 43))  # 43 Is the class for "no sign"

        shuffle(no_signs)
        return no_signs
    def __load_next_window(self):
        """
        load the next window. If the current image contains no more windows, loads the next window.
        :return: A window and the corresponding list of regions of interest
        :returns: list[list[list[float]]], int
        """
        try:
            image_raw, current_window = self._sliding_window.next()
        except:
            self._image, image_data = self.__load_next_image()
            self._sliding_window = ScalingSlidingWindow(image_data, self.window_size, 1,
                                                        zoom_factor=lambda x: 1 - 0.005 * x, overlap=.9)
            image_raw, current_window = self._sliding_window.next()

        # Find all regions of interest, that overlap to at lest 50% with this region of interest
        regions = self._image.get_overlapping_regions(current_window, 0.50)

        # label 43 is used for "no sign"
        # Load and prepare ground truth
        label = 43

        if len(regions) == 1:
            label = regions[0]

        if False and len(regions) > 0:
            print "({},{}), ({},{})".format(regions[0].x1, regions[0].y1, regions[0].x2, regions[0].y2)
            print "({},{}), ({},{})".format(current_window.x1, current_window.y1, current_window.x2, current_window.y2)
            print "Overlap: {}".format(regions[0].get_overlap(current_window))
            show_image_and_label(image_raw, label)

        return image_raw, label

    def __load_next_image(self):
        """
        Load the next image.

        :return: Returns the next labeled image and the raw image data
        :returns: IdentifiedImage, list[list[list[float]]]
        """

        # Did we finish an epoch?
        if self._cur == len(self.images):
            self._cur = 0
            shuffle(self.images)

        # Load an image
        image = self.images[self._cur]
        image_data = np.asarray(Image.open(osp.join(self.gtsdb_root, 'JPEGImages', image.path)))

        # show_image_and_regions(image_data, image.region_of_interests)

        self._cur += 1
        return image, image_data


def get_images_and_regions(gtsdb_root, min=0, max=599):
    """
    Generates a list of images in the GTSDB contest. Only a maximum of 600 images are used. This is necessary, to also
    be able to use just the training data set

    :param gtsdb_root: The root of image folder. In this folder all the images have to reside.
    :return: returns a list of identified images. These are already labeled
    :returns: list[IdentifiedImage]
    :type gtsdb_root: str
    """
    # get list of image indexes.
    duplicated_images = []  # images
    with open(gtsdb_root + "/gt.txt") as gtFile:
        gt_reader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        # loop over all images in current annotations file
        for row in gt_reader:
            path_to_image = gtsdb_root + "/" + row[0]

            # find size of image
            roi = [RegionOfInterest(row[1], row[2], row[3], row[4], row[5])]
            image = IdentifiedImage(path_to_image, roi)
            duplicated_images.append(image)

    # fill in all the images without a region of interest
    # there are images ranging from 00000 to 00599
    # higher numbers are ignored.
    for i in range(min, max):
        found = False
        for image in duplicated_images:
            if str(i) in image.path:
                found = True
                break

        if not found:
            duplicated_images.append(IdentifiedImage(gtsdb_root + '/' + format(i, '05d') + '.ppm', []))

    # loop over all images to make sure, that no duplicate file path exists
    images = []
    last_image = duplicated_images[0]
    images.append(last_image)
    for image in duplicated_images[0:]:
        if last_image.path == image.path:
            rois = last_image.region_of_interests
            rois.extend(image.region_of_interests)
            last_image.region_of_interests = rois
        else:
            last_image = image
            images.append(last_image)
    shuffle(images)
    return images


def show_image_and_label(image, label_info):
    from scipy.misc import imshow
    print label_info
    imshow(image)


def show_image_and_regions(image, regions):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from skimage.measure import label
    label_image = label(image.copy())

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(label_image)

    for region in regions:
        rect = mpatches.Rectangle((region.x1, region.y1), region.x2 - region.x1, region.y2 - region.y1,
                                  fill=False, edgecolor='red', linewidth=2)

        ax.add_patch(rect)

    plt.show()
