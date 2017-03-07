import abc

import caffe


class InputLayer(caffe.Layer):
    """
    Input layer for bb net. Has dynamic input size and fixed output size. Input is a proposed region of a
    activation layer. output should be

    On each forward, the layer should reshape the top layer.
    """

    @property
    @abc.abstractproperty
    def default_shape_data(self):
        return []

    @property
    @abc.abstractproperty
    def default_shape_label(self):
        return []

    def __init__(self, p_object, *args, **kwargs):
        super(InputLayer, self).__init__(p_object, *args, **kwargs)

    def setup(self, bottom, top):
        # Warn, if this layer got an input. It will be ignored.
        if len(bottom) > 0:
            print "Warning: Input layer will ignore bottom layers."

        # Parse input arguments. These come from the net prototxt model
        args = parse_arguments(self.param_str)
        self.apply_arguments(args)

        # Initial shaping is needed
        top[0].reshape(*self.default_shape_data)
        top[1].reshape(*self.default_shape_label)

    @abc.abstractmethod
    def apply_arguments(self, args):
        pass

    @abc.abstractmethod
    def get_next_data(self):
        return None, None

    def forward(self, bottom, top):
        # 1. Get new data to use
        net_data, label_data = self.get_next_data()

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
