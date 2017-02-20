from random import shuffle

import caffe

from sign_detection.GTSDB.SlidingWindow.batchloader import BatchLoader
from sign_detection.GTSRB.ImageReader import read_test_traffic_signs, read_train_traffic_signs


class VaryingSizeDataLayer(caffe.Layer):
    """
    This is a layer for training the detection net. The net returns 1 if exactly one traffic sign is in a
    region of interest.
    """

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        self.params = params
        check_params(params)

        self.batch_size = params['batch_size']

        # load the images
        if (params['gtsrb_train_root'] != 'null'):
            images = read_train_traffic_signs(params['gtsrb_train_root'])
        else:
            images = read_test_traffic_signs(params['gtsrb_test_root'])

        # shuffle the images
        shuffle(images)

        self.images_and_labels = []

        for image in self.images:
            im = caffe.io.load_image(image.path_to_image)
            self.images_and_labels.append((im, image.get_region_of_interests[0].sign))

        # remember indices
        self.current_index = 0

        # Use one values to determine the class, if a region contains a sign class = 1, else 0
        top[1].reshape(self.batch_size, 43)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            im, label = self.images_and_labels[self.current_index]
            self.current_index += 1
            if self.current_index >= len(self.images_and_labels):
                self.current_index = 0
                shuffle(self.images_and_labels)

            top[0].reshape(self.batch_size, 3, im.shape[1], im.shape[2])

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


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['gtsrb_train_root', 'gtsrb_test_root', 'batch_size']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
