# imports
import scipy.misc
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from PIL import Image
import csv

from sign_detection.model import SlidingWindow
from sign_detection.model.IdentifiedImage import IdentifiedImage
from sign_detection.model.RegionOfInterest import RegionOfInterest


class GtsdbSlidingWindowDataLayer(caffe.Layer):
    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        check_params(params)

        self.batch_size = params['batch_size']
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 43 channels (because GTSDB has 43 classes.)
        top[1].reshape(self.batch_size, 43)

        print_info("GtsrbSlidingWindowDataLayer", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_window()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

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
        self.im_shape = params['im_shape']
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

    def load_next_window(self):
        if (self._sliding_window == None):
            self._image = self.load_next_image()
            self._sliding_window = SlidingWindow.SlidingWindow(image=self._image, size=0.01)

        # Create a region of interest
        # ToDO Check if this is a method of the SlidingWindow class
        x1 = self._sliding_window.window_pos[0]
        x2 = self._sliding_window.window_pos[0] + self._sliding_window.window_size[0]
        y1 = self._sliding_window.window_pos[1]
        y2 = self._sliding_window.window_pos[1] + self._sliding_window.window_size[1]

        current_window = RegionOfInterest(x1, y1, x2, y2, sign=0)

        # Find all regions of interest, that overlap to at lest 90% with this region of interest
        regions = self._image.get_overlapping_regions(current_window, 0.9)

        # Load and prepare ground truth
        multilabel = np.zeros(43).astype(np.float32)
        for roi in regions:
            multilabel[int(roi.sign) - 1] = 1

        # ToDo Catch exception of iter and instead return the next image.
        return self._sliding_window.__iter__

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
        image_data = scipy.misc.imresize(image_data, self.im_shape)  # resize

        self._cur += 1
        return preprocess(image_data)


def preprocess(im):
    """
    preprocess() emulate the pre-processing occuring in the vgg16 caffe
    prototxt.
    """

    im = np.float32(im)
    im = im[:, :, ::-1]  # change to BGR
    im = im.transpose((2, 0, 1))

    return im

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['batch_size', 'gtsdb_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized with bs: {}, im_shape: {}.".format(
        name,
        params['batch_size'],
        params['im_shape'])
