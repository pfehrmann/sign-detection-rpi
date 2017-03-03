import caffe

import sign_detection.tools.batchloader as bl
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from sign_detection.model.IdentifiedImage import IdentifiedImage


class InputLayer(caffe.Layer):
    """
    Input layer for bb net. Has dynamic input size and fixed output size. Input is a proposed region of a
    activation layer. output should be

    On each forward, the layer should reshape the top layer.
    """

    def __init__(self, p_object, *args, **kwargs):
        super(InputLayer, self).__init__(p_object, *args, **kwargs)

        # Init class variables
        self.file_input_net = ''  # type: str
        self.file_input_weights = ''  # type: str
        self.location_gt = ''  # type: str
        self.shape = [1, 64, 20, 20]  # type: list
        self.images = []  # type: list
        self.image_iterator = iter([])  # type: listiterator
        self.rois = iter([])  # type: listiterator
        self.input_detector = None  # type: un.Detector
        self.net = None  # type: caffe.Net

    def setup(self, bottom, top):
        self.top_names = ['activation', 'label']  # Do we need that?

        # Setup validation
        if len(bottom) > 0:
            raise Exception('Input layer cannot have other layers as input.')

        self.parse_arguments(self.param_str)

        self.load_input_detector()

        self.load_images()

    def forward(self, bottom, top):
        """
        1. Get new data to use
        2. Push activation data into net
        3. Push label data into loss (the ground truth roi)
        """
        # 1. Get new data to use
        roi, activation = self.get_next_data()

        # 2.
        self.shape = activation.shape
        self.reshape(None, top)
        top[0].data[...] = activation

        # 3.
        top[1].data[...] = roi.get_vector()

    def reshape(self, bottom, top):
        if self.shape is not None:
            top[0].reshape(*self.shape)

        top[1].reshape(1, 4)

    def backward(self, top, propagate_down, bottom):
        """Input layer does not back propagate"""

    def get_next_data(self):
        try:
            roi = self.rois.next()
            image = roi[0]
            region = roi[1]
            activation = self.calculate_activation(image)
            return region, activation
        except StopIteration:
            try:
                img = self.image_iterator.next()
                img_raw = caffe.io.load_image(img.path)
                rois = [roi.clone()
                        .disturb()
                        .add_padding(11)
                        .ensure_bounds(max_x=len(img_raw[0]), max_y=len(img_raw))
                        for roi in img.get_region_of_interests()]
                cropped_rois = [(img_raw[roi.y1:roi.y2, roi.x1:roi.x2, :], roi) for roi in rois]
                self.rois = iter(cropped_rois)
                return self.get_next_data()
            except StopIteration:
                self.image_iterator = iter(self.images)
                return self.get_next_data()

    def calculate_activation(self, img):
        return self.input_detector.get_activation(img)

    def parse_arguments(self, arg_string):
        if not isinstance(arg_string, str) or not arg_string:
            raise Exception('Input layer: No params given. please specify param_str in your net definition (prototxt)')
        params = eval(arg_string)
        self.file_input_net = parse_arg(params, 'file_input_net', str)
        self.file_input_weights = parse_arg(params, 'file_input_weights', str)
        self.location_gt = parse_arg(params, 'location_gt', str)

    def load_input_detector(self):
        net = un.load_net(self.file_input_net, self.file_input_weights)
        self.input_detector = un.Detector(net, minimum=0.9999, use_global_max=False, threshold_factor=0.75,
                                          draw_results=False,
                                          zoom=[1], area_threshold_min=1200, area_threshold_max=50000,
                                          activation_layer="conv3",
                                          out_layer="softmax", display_activation=False, blur_radius=1, size_factor=0.5,
                                          faster_rcnn=True, modify_average_value=True, average_value=30)

    def load_images(self):
        self.images = bl.get_images_and_regions(self.location_gt)
        # Create 'plain' objects: each image should only contain one roi


def parse_arg(params, arg, arg_type):
    if arg not in params:
        raise Exception('Input layer: Missing setup param "{0}"' % arg)
    val = params[arg]
    if not isinstance(val, arg_type):
        raise Exception('Input layer: Setup param "{0}" has invalid type' % arg)
    return val
