import caffe

from pydoc import locate

# TODO should be moved to input arguments
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

        # Import and create data source
        data_class = locate(args['data_source_class'])
        self.data_source = data_class(args)

        # Initial shaping is needed
        top[0].reshape(*default_shape_data)
        top[1].reshape(*default_shape_label)

    def forward(self, bottom, top):
        # 1. Get new data to use
        net_data, label_data = self.data_source.get_next_data()

        # 2. Reshape the net and then push data into it
        top[0].reshape(*net_data.shape)
        top[0].data[...] = net_data

        # 3. Push label data into loss
        top[1].data[...] = label_data

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
