import csv
from os import path as osp
from random import shuffle

import numpy as np
from PIL import Image

from sign_detection.model.IdentifiedImage import IdentifiedImage
from sign_detection.model.RegionOfInterest import RegionOfInterest


class BatchLoader(object):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    :ivar _image: IdentifiedImage
    """

    def __init__(self, params):
        """
        Initialize the batch loader. This already reads the list of images but generates no windows.
        :param params: The parameter dictionary containing the values batch_size, gtsdb_root and window_size
        :param num: The number of windows to go through before filtering some out. Note that the higher this number is,
                    the more random the results are but also more RAM will be used.
        :param fraction: The percentage of images with signs (eg. fraction == 0.3: signs: 100, no signs: 233)
        """
        self.gtsdb_root = params['gtsdb_root']
        self._cur = 0
        self._image = None

        self.images = get_images_and_regions(self.gtsdb_root)

        print "BatchLoader initialized with %s images" % len(self.images)

    def load_next_image(self):
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
        image_data = np.asarray(Image.open(osp.join(self.gtsdb_root, 'JPEGImages', image.path))).transpose(2, 1, 0)

        self._cur += 1
        return image, image_data


def get_images_and_regions(gtsdb_root):
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
