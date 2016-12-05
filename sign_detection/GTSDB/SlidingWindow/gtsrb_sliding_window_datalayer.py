import caffe
import numpy as np

from sign_detection.GTSDB.SlidingWindow.batchloader import BatchLoader


class GtsdbSlidingWindowDataLayer(caffe.Layer):
    """
    This is a layer for training the detection net. The net returns 1 if exactly one traffic sign is in a region of interest.
    """

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        check_params(params)

        self.batch_size = params['batch_size']
        self.batch_loader = BatchLoader(params, 150000, 0.33)

        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, 3, params['window_size'], params['window_size'])

        # Use two values to determine, if a region contains a sign ([1]=1) or not ([0]=1)
        top[1].reshape(self.batch_size, 2)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.batch_loader.next_window()

            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


def preprocess(im):
    """
    preprocess() emulate the pre-processing occuring in the vgg16 caffe
    prototxt.
    """

    im = np.float32(im)
    #im = im[:, :, ::-1]  # change to BGR
    #im = im.transpose((2, 0, 1))

    return im


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['batch_size', 'gtsdb_root', 'window_size']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
