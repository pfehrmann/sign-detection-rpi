from random import shuffle

import caffe

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import sign_detection.tools.batchloader as bl


class ActivationMapSource:
    def __init__(self, args):

        # Init vars
        self.images = []  # type: list
        self.image_current = -1  # type: int
        self.image_max = -1  # type: int
        self.input_detector = None  # type: un.Detector

        # Parse args
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

        # Modify image
        modified_roi = roi.clone().disturb().ensure_bounds(max_x=len(img_raw[0]), max_y=len(img_raw))
        image_excerpt = img_raw[modified_roi.y1:modified_roi.y2, modified_roi.x1:modified_roi.x2, :]

        # Create loss vector
        d1 = (modified_roi.p1 - roi.p1).as_array
        d2 = (modified_roi.p2 - roi.p2).as_array
        v = d1 + d2

        return self.calculate_activation(image_excerpt), v

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
        raise Exception('Input layer: Missing setup param "{0}"' % arg)
    val = params[arg]
    if not isinstance(val, arg_type):
        raise Exception('Input layer: Setup param "{0}" has invalid type' % arg)
    return val
