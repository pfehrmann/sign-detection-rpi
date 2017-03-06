from random import shuffle

import caffe

from sign_detection.GTSDB.BoundingBoxRegression.input_data import InputData
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import sign_detection.tools.batchloader as bl


class ActivationMapSource:

    def __init__(self, args):

        # Init vars
        self.images = []  # type: list
        self.image_current = -1  # type: int
        self.image_max = -1  # type: int
        self.data = []  # type: list
        self.data_current = -1  # type: int
        self.data_max = -1  # type: int
        self.input_detector = None  # type: un.Detector

        # Parse args
        self.file_input_net = parse_arg(args, 'file_input_net', str)
        self.file_input_weights = parse_arg(args, 'file_input_weights', str)
        self.location_gt = parse_arg(args, 'location_gt', str)

        # Load data
        self.load_input_detector()
        self.load_images()

    def get_next_data(self):
        self.data_current += 1
        if self.data_current >= self.data_max:
            self.load_next_image()
            return self.get_next_data()
        return self.data[self.data_current]

    def load_next_image(self):
        self.image_current += 1
        if self.image_current >= self.image_max:
            self.image_current = 0
            shuffle(self.images)
            # TODO what if 0 images there?

        img_inf = self.images[self.image_current]
        img_raw = caffe.io.load_image(img_inf.path)

        rois = [roi.clone()
                .disturb()
                .add_padding(11)
                .ensure_bounds(max_x=len(img_raw[0]), max_y=len(img_raw))
                for roi in img_inf.get_region_of_interests()]

        self.data = [InputData(self.calculate_activation(img_raw[roi.y1:roi.y2, roi.x1:roi.x2, :]), roi.get_vector())
                     for roi in rois]
        self.data_max = len(self.data)
        self.data_current = -1

    def calculate_activation(self, img):
        return self.input_detector.get_activation(img)

    def load_images(self):
        image_info_list = bl.get_images_and_regions(self.location_gt)
        self.images = image_info_list
        self.image_max = len(self.images)

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
        raise Exception('Input layer: Missing setup param "{0}"' % arg)
    val = params[arg]
    if not isinstance(val, arg_type):
        raise Exception('Input layer: Setup param "{0}" has invalid type' % arg)
    return val
