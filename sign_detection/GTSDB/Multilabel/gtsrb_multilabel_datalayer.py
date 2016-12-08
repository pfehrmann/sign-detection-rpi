import caffe

from sign_detection.GTSDB.Multilabel.batchloader import BatchLoader



class GtsdbMultilabelDataLayer(caffe.Layer):
    """

    """

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        check_params(params)

        self.batch_size = params['batch_size']
        self.batch_loader = BatchLoader(params)

        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])

        top[1].reshape(self.batch_size, 43)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, im_raw = self.batch_loader.load_next_image()

            v = get_class_label_vector(map(lambda roi: int(roi.sign), im.get_region_of_interests()))

            top[0].data[itt, ...] = im_raw
            top[1].data[itt, ...] = v

    def reshape(self, Layer, *args, **kwargs):
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

    required = ['batch_size', 'gtsdb_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include %s' % r


def get_class_label_vector(classes):
    label_vector = [False] * 43
    for i in classes:
        label_vector[i] = True
    return label_vector
