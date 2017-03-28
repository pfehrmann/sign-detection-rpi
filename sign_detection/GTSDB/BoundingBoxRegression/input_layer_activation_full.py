import numpy
from random import shuffle

import caffe
import cv2

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
        location_cache = parse_arg(args, 'location_activation_cache', str)
        self.activation_cache = ActivationCache(location_cache, file_input_net, file_input_weights, load_image,
                                                ignore_persistence=True)
        self.location_gt = parse_arg(args, 'location_gt', str)

        # Load data
        self.load_activations()
        # self.activation_cache.free_memory() # only do, if activations are calculated before

    def get_next_data(self):
        # Increase image counter
        self.image_current += 1
        if self.image_current >= self.image_max:
            self.image_current = 0
            shuffle(self.activations)

        # Get net image
        img, region = self.activations[self.image_current]
        activation, gt_roi = self.get_activation_and_scale_roi(img, region)  # region.add_padding(11)

        # Calculate activation
        mod_roi = gt_roi.clone().disturb().clip(max_x=activation.shape[3], max_y=activation.shape[2])
        image_excerpt = activation[:, :, mod_roi.y1:mod_roi.y2, mod_roi.x1:mod_roi.x2]

        # Create loss vector
        v = loss_vector(mod_roi, gt_roi)

        return image_excerpt, v

    def load_activations(self):
        # Get image info
        image_info_list = bl.get_images_and_regions(self.location_gt)
        print 'Got {0} images to work with. Loading activation maps.'.format(len(image_info_list))

        # Read or calculate activation maps for each roi
        self.activations = []
        for img in image_info_list:
            for region in img.region_of_interests:
                self.activations.append((img, region))
        shuffle(self.activations)

        # Check, if images are there
        self.image_max = len(self.activations)
        if self.image_max < 1:
            raise Exception('No images for training found.')
        print "Got {0} ROIs.".format(self.image_max)

    def get_activation_and_scale_roi(self, img_info, roi):
        """Either calculates or loads the activation map for the given image using the activation cache.
        Furthermore creates an roi that contains information, how much the activation map differs from the image."""

        activation, factors = self.activation_cache.load(img_info)
        roi_scaled = scaled_roi(roi, factors[0], factors[1], probability=1)
        return activation, roi_scaled


def parse_arg(params, arg, arg_type):
    if arg not in params:
        raise Exception('ActivationMapSource: Missing setup param "{0}"' % arg)
    val = params[arg]
    if not isinstance(val, arg_type):
        raise Exception('ActivationMapSource: Setup param "{0}" has invalid type' % arg)
    return val


dict = {}
def load_image(path):
    global dict
    if dict.has_key(path):
        # convert back to 32 bit to ensure compatability with caffe # todo check if this is needed
        return numpy.float32(dict[path])
    img_raw = caffe.io.load_image(path)

    # as the net is trained with digits, images have to range between 0 and 255
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB) * 255.0

    # store in dict as a cache. Convert to 8 bit to save space
    dict[path] = numpy.int8(img_raw)
    return img_raw


def loss_vector(mod_roi, gt_roi):
    gt_ctr = gt_roi.center
    mod_ctr = mod_roi.center
    targets_dx = (gt_ctr.x - mod_ctr.x) / (mod_roi.width + 1.0)
    targets_dy = (gt_ctr.y - mod_ctr.y) / (mod_roi.height + 1.0)
    targets_dw = numpy.log(gt_roi.width / (mod_roi.width + 1.0))
    targets_dh = numpy.log(gt_roi.height / (mod_roi.height + 1.0))

    targets = numpy.array([targets_dx, targets_dy, targets_dw, targets_dh])

    return targets
