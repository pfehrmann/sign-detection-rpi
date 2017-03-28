import numpy
from random import shuffle

import caffe
import cv2
import hashlib
import os

import sign_detection.tools.batchloader as bl
from sign_detection.GTSDB.BoundingBoxRegression.activation_cache import ActivationCache
from sign_detection.GTSDB.BoundingBoxRegression.input_layer import InputLayer
from sign_detection.model.PossibleROI import scaled_roi



class InputLayerActivationFull(InputLayer):
    @property
    def default_shape_label(self):
        return [1, 4]

    @property
    def default_shape_data(self):
        return [1, 64, 2, 2]

    def __init__(self, p_object, *args, **kwargs):
        super(InputLayerActivationFull, self).__init__(p_object, *args, **kwargs)

        # Init vars
        self.activations = []  # type: list
        self.image_current = -1  # type: int
        self.image_max = -1  # type: int
        self.location_gt = ''  # type: str
        self.activation_cache = None  # type: ActivationCache

    def apply_arguments(self, args):
        file_input_net = parse_arg(args, 'file_input_net', str)
        file_input_weights = parse_arg(args, 'file_input_weights', str)
        self.activation_cache = ActivationCache("data/activation/", file_input_net, file_input_weights)
        self.location_gt = parse_arg(args, 'location_gt', str)

        # Load data
        self.load_activations()
        self.activation_cache.free_memory()

    def get_next_data(self):
        # Increase image counter
        self.image_current += 1
        if self.image_current >= self.image_max:
            self.image_current = 0
            shuffle(self.activations)

        # Get net image
        activation, gt_roi = self.activations[self.image_current]

        # Calculate activation
        mod_roi = gt_roi.clone().disturb().clip(max_x=activation.shape[3], max_y=activation.shape[2])
        image_excerpt = activation[:, :, mod_roi.y1:mod_roi.y2, mod_roi.x1:mod_roi.x2]

        # Create loss vector
        v = loss_vector(mod_roi, gt_roi)

        return image_excerpt, v

    def load_activations(self):
        # Get image info
        image_info_list = bl.get_images_and_regions(self.location_gt)[:50]
        print 'Got {0} images to work with. Loading activation maps.'.format(len(image_info_list))

        # Read or calculate activation maps for each roi
        self.activations = [self.get_activation_and_scale_roi(img, region.add_padding(11))
                            for img in image_info_list for region in img.region_of_interests]
        shuffle(self.activations)

        # Check, if images are there
        self.image_max = len(self.activations)
        if self.image_max < 1:
            raise Exception('No images for training found.')
        print "Got {0} ROIs.".format(self.image_max)

    def get_activation_and_scale_roi(self, img_info, roi):
        img_name = os.path.basename(img_info.path)
        if self.activation_cache.has(img_name):
            activation, factors = self.activation_cache.load(img_name)
        else:
            img_data = load_image(img_info.path)
            activation, factors = self.activation_cache.add(img_data, img_name)
        return activation, scaled_roi(roi, factors[0], factors[1], probability=1)


def parse_arg(params, arg, arg_type):
    if arg not in params:
        raise Exception('ActivationMapSource: Missing setup param "{0}"' % arg)
    val = params[arg]
    if not isinstance(val, arg_type):
        raise Exception('ActivationMapSource: Setup param "{0}" has invalid type' % arg)
    return val


def load_image(path):
    img_raw = caffe.io.load_image(path)
    return cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)


def loss_vector(mod_roi, gt_roi):
    gt_ctr = gt_roi.center
    mod_ctr = mod_roi.center
    targets_dx = (gt_ctr.x - mod_ctr.x) / (mod_roi.width + 1.0)
    targets_dy = (gt_ctr.y - mod_ctr.y) / (mod_roi.height + 1.0)
    targets_dw = numpy.log(gt_roi.width / (mod_roi.width + 1.0))
    targets_dh = numpy.log(gt_roi.height / (mod_roi.height + 1.0))

    targets = numpy.array([targets_dx, targets_dy, targets_dw, targets_dh])

    return targets