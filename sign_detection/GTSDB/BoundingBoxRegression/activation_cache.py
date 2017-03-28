import os
import numpy as np
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un


class ActivationCache:
    def __init__(self, path_cache, file_input_net, file_input_weights, load_image, detector_config=None):
        """Creates a new instance of a activation cache.
        :param path_cache The file path where the cached activation maps are stored.
        :param file_input_net the location of the net that calculates the activation maps.
        :param file_input_weights the location of the weight for the activation net.
        :param load_image a function that takes an image path and returns the raw image data for the detector.
        :param detector_config a object for configuring the activation detector.
        """

        self.detector_config = {"minimum": 0.9999, "use_global_max": False, "threshold_factor": 0.75,
                                "draw_results": False, "zoom": [1], "area_threshold_min": 1200,
                                "area_threshold_max": 50000, "activation_layer": "conv3", "out_layer": "softmax",
                                "display_activation": False, "blur_radius": 1, "size_factor": 0.5, "faster_rcnn": True,
                                "modify_average_value": True, "average_value": 30}
        if detector_config is not None:
            self.detector_config.update(detector_config)
        self.path = path_cache
        self.can_calculate = False
        self.input_detector = None  # type: un.Detector
        self.input_net = file_input_net
        self.input_weights = file_input_weights
        self.load_image = load_image

    def load(self, img):
        img_name = os.path.basename(img.path)
        if not self.__has(img_name):
            return self.__add(img)
        data = np.load(self.path + img_name)
        return data[0], data[1]

    def free_memory(self):
        self.__disable_calculating()

    def __has(self, img_name):
        return os.path.isfile(self.path + img_name)

    def __add(self, img):
        if not self.can_calculate:
            self.__enable_calculating()
        img_data = self.load_image(img.path)
        activation = self.input_detector.calculate_activation(img_data)
        factors = [activation.shape[3] / float(img_data.shape[1]),
                   activation.shape[2] / float(img_data.shape[0])]
        data = [activation, factors]
        np.save(self.path + os.path.basename(img.path) + ".npy", data)
        return activation, factors

    def __enable_calculating(self):
        self.__load_input_detector()
        self.can_calculate = True

    def __disable_calculating(self):
        self.can_calculate = False
        del self.input_detector.net
        del self.input_detector

    def __load_input_detector(self):
        net = un.load_net(self.input_net, self.input_weights)
        self.input_detector = un.Detector(net, **self.detector_config)
