from random import shuffle

import caffe
import cv2

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import sign_detection.tools.batchloader as bl
from sign_detection.GTSDB.BoundingBoxRegression.input_layer import InputLayer


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
        self.images = []  # type: list
        self.image_current = -1  # type: int
        self.image_max = -1  # type: int
        self.input_detector = None  # type: un.Detector

        self.file_input_net = ''  # type: str
        self.file_input_weights = ''  # type: str
        self.location_gt = ''  # type: str

    def apply_arguments(self, args):
        self.file_input_net = parse_arg(args, 'file_input_net', str)
        self.file_input_weights = parse_arg(args, 'file_input_weights', str)
        self.location_gt = parse_arg(args, 'location_gt', str)

        # Load data
        self.load_input_detector()
        self.load_images()

    def get_next_data(self):
        # Increase image counter
        self.image_current += 1
        if self.image_current >= self.image_max:
            self.image_current = 0
            shuffle(self.images)

        # Get net image
        path, roi = self.images[self.image_current]
        img_raw = caffe.io.load_image(path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # Calculate activation
        activation = self.calculate_activation(img_raw)
        factor_x = activation.shape[3] / float(img_raw.shape[1])
        factor_y = activation.shape[2] / float(img_raw.shape[0])
        modified_roi = roi.clone().scale(factor_x, factor_y).disturb().ensure_bounds(max_x=len(img_raw[0]), max_y=len(img_raw))
        image_excerpt = activation[:, :, int(modified_roi.y1):int(modified_roi.y2), int(modified_roi.x1):int(modified_roi.x2)]

        # Create loss vector
        d1 = (modified_roi.p1 - roi.clone().scale(factor_x, factor_y).p1).as_array
        d2 = (modified_roi.p2 - roi.clone().scale(factor_x, factor_y).p2).as_array
        v = d1 + d2

        if False:
            print 'DATA INFO:'
            print 'ORI ROI: %s' % str(roi)
            print 'MOD ROI: %s' % str(modified_roi)
            print 'COR VEC: %s' % str(v)

        return image_excerpt, v

    def calculate_activation(self, img):
        return self.input_detector.get_activation(img)

    def load_images(self):
        image_info_list = bl.get_images_and_regions(self.location_gt)
        self.images = [(img.path, region.add_padding(11))
                       for img in image_info_list for region in img.region_of_interests]
        shuffle(self.images)
        self.image_max = len(self.images)
        if self.image_max < 1:
            raise Exception('No images for training found.')
        print "Got {0} Boxes.".format(self.image_max)

    def load_input_detector(self):
        net = un.load_net(self.file_input_net, self.file_input_weights)
        self.input_detector = un.Detector(net, minimum=0.9999, use_global_max=False, threshold_factor=0.75,
                                          draw_results=False,
                                          zoom=[1], area_threshold_min=1200, area_threshold_max=50000,
                                          activation_layer="conv3",
                                          out_layer="softmax", display_activation=False, blur_radius=1, size_factor=0.5,
                                          faster_rcnn=True, modify_average_value=True, average_value=30)


def parse_arg(params, arg, arg_type):
    if arg not in params:
        raise Exception('ActivationMapSource: Missing setup param "{0}"' % arg)
    val = params[arg]
    if not isinstance(val, arg_type):
        raise Exception('ActivationMapSource: Setup param "{0}" has invalid type' % arg)
    return val
