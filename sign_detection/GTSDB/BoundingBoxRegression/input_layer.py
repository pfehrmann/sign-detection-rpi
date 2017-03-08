import caffe

from sign_detection.GTSDB.BoundingBoxRegression.activation_map_source import ActivationMapSource


class InputLayer(caffe.Layer):
    """
    Input layer for bb net. Has dynamic input size and fixed output size. Input is a proposed region of a
    activation layer. output should be

    On each forward, the layer should reshape the top layer.
    """

    def __init__(self, p_object, *args, **kwargs):
        super(InputLayer, self).__init__(p_object, *args, **kwargs)

        # Init class variables
        # TODO is this a good default shape? It will be used at least once.
        self.shape = [1, 64, 2, 2]  # type: list
        self.data_source = None
        self.net = None  # type: caffe.Net

    def setup(self, bottom, top):
        # Warn, if this layer got an input. It will be ignored.
        if len(bottom) > 0:
            print "Warning: Input layer will ignore bottom layers."

        # Parse input arguments. These come from the net prototxt model
        args = parse_arguments(self.param_str)

        # Create source class
        self.data_source = ActivationMapSource(args)

        top[1].reshape(1, 4)
        top[0].reshape(*self.shape)

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
