import caffe

from sign_detection.GTSDB.BoundingBoxRegression.activation_map_source import ActivationMapSource

# TODO is this a good default shape? It will be used at least once.
default_shape_data = [1, 64, 2, 2]
default_shape_label = [1, 4]


class InputLayer(caffe.Layer):
    """
    Input layer for bb net. Has dynamic input size and fixed output size. Input is a proposed region of a
    activation layer. output should be

    On each forward, the layer should reshape the top layer.
    """

    def __init__(self, p_object, *args, **kwargs):
        super(InputLayer, self).__init__(p_object, *args, **kwargs)

        # Init class variables
        self.data_source = None

    def setup(self, bottom, top):
        # Warn, if this layer got an input. It will be ignored.
        if len(bottom) > 0:
            print "Warning: Input layer will ignore bottom layers."

        # Parse input arguments. These come from the net prototxt model
        args = parse_arguments(self.param_str)

        # Create class that generates the data blobs
        self.data_source = ActivationMapSource(args)

        # Initial shaping is needed
        top[0].reshape(*default_shape_data)
        top[1].reshape(*default_shape_label)

    def forward(self, bottom, top):
        # 1. Get new data to use
        data = self.data_source.get_next_data()

        # 2. Reshape the net and then push data into it
        top[0].reshape(*data.net_data.shape)
        top[0].data[...] = data.net_data

        # 3. Push label data into loss
        top[1].data[...] = data.loss_data

    def reshape(self, bottom, top):
        """Reshaping is done manually"""
        pass

    def backward(self, top, propagate_down, bottom):
        """Input layer does not back propagate"""
        pass


def parse_arguments(arg_string):
    if not isinstance(arg_string, str) or not arg_string:
        return {}
    return eval(arg_string)
