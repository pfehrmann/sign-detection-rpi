import csv
import random
from os import path as osp
from random import shuffle

import numpy as np
from PIL import Image

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
        self.batch_size = params['batch_size']
        self.gtsdb_root = params['gtsdb_root']
        self.window_size = params['window_size']
        self._cur = 0
        self._sliding_window = None
        self._image = None
        self.num = num
        self.fraction = fraction

        self.images = get_images_and_regions(self.gtsdb_root)

        print "BatchLoader initialized with {} images".format(
            len(self.images))

    def next_window(self):
        try:
            return self._windows.pop()
        except:
            self.collect_windows()
            return self._windows.pop()

    def collect_windows(self):
        signs = []
        no_signs = []

        for i in range(0, self.num, 1):
            image, label = self.__load_next_window()
            if label[0] == 1:
                no_signs.append((image, label))
            else:
                signs.append((image, label))

        no_sign_count = max(len(signs) * self.fraction, len(no_signs))
        random.shuffle(no_signs)

        # create result array
        result = []
        result.extend(signs)
        result.extend(no_signs[0:int(no_sign_count)])
        random.shuffle(result)
        self._windows = result

    def __load_next_window(self):
        try:
            image_raw, current_window = self._sliding_window.next()
        except:
            self._image, image_data = self.load_next_image()
            self._sliding_window = ScalingSlidingWindow(image_data, self.window_size, 1,
                                                        zoom_factor=lambda x: 1 / (x + 1))
            image_raw, current_window = self._sliding_window.next()

        # Find all regions of interest, that overlap to at lest 90% with this region of interest
        regions = self._image.get_overlapping_regions(current_window, 0.85)

        # Load and prepare ground truth
        label = np.zeros(2).astype(np.float32)
        if len(regions) == 1:
            label[1] = 1
        else:
            label[0] = 1

        return image_raw, label

    def load_next_image(self):
        """
        Load the next image in a batch.
        """

        # Did we finish an epoch?
        if self._cur == len(self.images):
            self._cur = 0
            shuffle(self.images)

        # Load an image
        image = self.images[self._cur]
        image_data = np.asarray(Image.open(osp.join(self.gtsdb_root, 'JPEGImages', image.path)))

        self._cur += 1
        return image, image_data


def get_images_and_regions(gtsdb_root):
    # get list of image indexes.
    duplicated_images = []  # images
    with open(gtsdb_root + "/gt.txt") as gtFile:
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        # loop over all images in current annotations file
        for row in gtReader:
            path_to_image = gtsdb_root + "/" + row[0]

            # find size of image
            roi = [RegionOfInterest(row[1], row[2], row[3], row[4], row[5])]
            image = IdentifiedImage(path_to_image, roi)
            duplicated_images.append(image)

    # fill in all the images without a region of interest
    # there are images ranging from 00000 to 00599
    for i in range(0, 599):
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

    return images
