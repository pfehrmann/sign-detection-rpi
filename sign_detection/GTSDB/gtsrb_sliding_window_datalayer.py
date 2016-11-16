# imports
import scipy.misc
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from PIL import Image
import csv
from sign_detection.model.IdentifiedImage import IdentifiedImage
from sign_detection.model.RegionOfInterest import RegionOfInterest


class GtsdbSlidingWindowDataLayer(caffe.Layer):
    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the parameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 43 channels (because GTSDB has 20 classes.)
        top[1].reshape(self.batch_size, 43)

        print_info("GtsrbSlidingWindowDataLayer", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_image()

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
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.gtsdb_root = params['gtsdb_root']
        self.im_shape = params['im_shape']
        self._cur = 0

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
        for i in range(0, 899):
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

        # Load and prepare ground truth
        multilabel = np.zeros(43).astype(np.float32)
        for roi in image.region_of_interests:
            # in the multilabel problem we don't care how MANY instances
            # there are of each class. Only if they are present.
            # The "-1" is b/c we are not interested in the background
            # class.
            multilabel[int(roi.sign) - 1] = 1

        self._cur += 1
        return preprocess(image_data), multilabel


def preprocess(im):
    """
    preprocess() emulate the pre-processing occuring in the vgg16 caffe
    prototxt.
    """

    im = np.float32(im)
    im = im[:, :, ::-1]  # change to BGR
    im = im.transpose((2, 0, 1))

    return im

def load_pascal_annotation(index, pascal_root):
    """
    This code is borrowed from Ross Girshick's FAST-RCNN code
    (https://github.com/rbgirshick/fast-rcnn).
    It parses the PASCAL .xml metadata files.
    See publication for further details: (http://arxiv.org/abs/1504.08083).
    Thanks Ross!
    """
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(21)))

    filename = osp.join(pascal_root, 'Annotations', index + '.xml')

    # print 'Loading: {}'.format(filename)

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 21), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) - 1
        y1 = float(get_data_from_tag(obj, 'ymin')) - 1
        x2 = float(get_data_from_tag(obj, 'xmax')) - 1
        y2 = float(get_data_from_tag(obj, 'ymax')) - 1
        cls = class_to_ind[
            str(get_data_from_tag(obj, "name")).lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'index': index}


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
