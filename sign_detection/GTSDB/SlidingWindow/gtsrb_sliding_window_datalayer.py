import random
import caffe

import numpy as np
import os.path as osp

from random import shuffle
from PIL import Image
import csv

from sign_detection.model.IdentifiedImage import IdentifiedImage
from sign_detection.model.RegionOfInterest import RegionOfInterest
from sign_detection.model.ScalingSlidingWindow import ScalingSlidingWindow


class GtsdbSlidingWindowDataLayer(caffe.Layer):
    """
    This is a layer for training the detection net. The net returns 1 if exactly one traffic sign is in a region of interest.
    """

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        check_params(params)

        self.batch_size = params['batch_size']
        self.batch_loader = BatchLoader(params, None)

        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, 3, params['window_size'], params['window_size'])

        # Use two values to determine, if a region contains a sign ([1]=1) or not ([0]=1)
        top[1].reshape(self.batch_size, 2)

    def forward(self, bottom, top):
        """
        Load data.
        """
        regions = []
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.batch_loader.next_window(150000, 0.33)

            top[0].data[itt, ...] = regions[itt][0]
            top[1].data[itt, ...] = regions[itt][1]

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    :ivar _image: IdentifiedImage
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.gtsdb_root = params['gtsdb_root']
        self.im_shape = params['window_size']
        self._cur = 0
        self._sliding_window = None
        self._image = None

        # get list of image indexes.
        list_file = 'gt.txt'

        self.duplicated_images = []  # images
        gtFile = open(self.gtsdb_root + "/" + list_file)  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file

        # loop over all images in current annotations file
        for row in gtReader:
            path_to_image = self.gtsdb_root + "/" + row[0]

            # find size of image
            roi = [RegionOfInterest(row[1], row[2], row[3], row[4], row[5])]
            image = IdentifiedImage(path_to_image, roi)
            self.duplicated_images.append(image)
        gtFile.close()

        # fill in all the images without a region of interest
        for i in range(0, 599):
            found = False
            for image in self.duplicated_images:
                if (str(i) in image.path):
                    found = True
                    break

            if (not found):
                self.duplicated_images.append(IdentifiedImage(self.gtsdb_root + '/' + format(i, '05d') + '.ppm', []))

        # loop over all images to make sure, that no duplicat file path exists
        self.images = []
        last_image = self.duplicated_images[0]
        self.images.append(last_image)
        for image in self.duplicated_images[0:]:
            if (last_image.path == image.path):
                rois = last_image.region_of_interests
                rois.extend(image.region_of_interests)
                last_image.region_of_interests = rois
            else:
                last_image = image
                self.images.append(last_image)

        print "BatchLoader initialized with {} images".format(
            len(self.images))

    def next_window(self, num, fraction):
        try:
            return self._windows.pop()
        except:
            self.collect_windows(num, fraction)
            return self._windows.pop()

    def collect_windows(self, num, fraction):
        signs = []
        no_signs = []

        for i in range(0,num, 1):
            image, label = self.__load_next_window()
            if label[0] == 1:
                no_signs.append((image, label))
            else:
                signs.append((image, label))

        no_sign_count = len(signs)*fraction
        random.shuffle(no_signs)
        result = []
        result.extend(signs)
        result.extend(no_signs[0:int(no_sign_count)])
        random.shuffle(result)
        self._windows = result

    def __load_next_window(self):
        image_raw = None
        current_window = None
        try:
            image_raw, current_window = self._sliding_window.next()
        except:
            self._image, image_data = self.load_next_image()
            self._sliding_window = ScalingSlidingWindow(preprocess(image_data), 64, 1,
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
        image = self.images[self._cur]  # Get the image index
        image_data = np.asarray(Image.open(
            osp.join(self.gtsdb_root, 'JPEGImages', image.path)))
        # image_data = scipy.misc.imresize(image_data, self.im_shape)  # resize

        self._cur += 1
        return image, image_data


def preprocess(im):
    """
    preprocess() emulate the pre-processing occuring in the vgg16 caffe
    prototxt.
    """

    im = np.float32(im)
    #im = im[:, :, ::-1]  # change to BGR
    #im = im.transpose((2, 0, 1))

    return im


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['batch_size', 'gtsdb_root', 'window_size']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
