import unittest
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import caffe
import numpy as np

from sign_detection.model.RegionOfInterest import RegionOfInterest


def fun(x):
    return x + 1


class UseNetTest(unittest.TestCase):
    @staticmethod
    def lenet():
        # our version of LeNet: a series of linear and simple nonlinear transformations
        n = caffe.NetSpec()
        return n

    def setUp(self):
        net = self.lenet()
        self.detector = un.Detector(net,
                                    minimum=0.25,
                                    use_global_max=False,
                                    threshold_factor=0.75,
                                    draw_results=False,
                                    zoom=[1, 2],
                                    area_threshold_min=1000,
                                    area_threshold_max=500000,
                                    activation_layer="activation",
                                    out_layer="softmax",
                                    global_pooling_layer="conv3",
                                    display_activation=False,
                                    blur_radius=1,
                                    size_factor=0.5,
                                    max_overlap=0.0,
                                    faster_rcnn=True,
                                    modify_average_value=True,
                                    average_value=100)

    def test(self):
        print("Testing filter regions...")
        activation_map = np.zeros((1, 1, 64, 1))
        rois = [
            (RegionOfInterest(1, 1, 10, 10, -1), activation_map),
            (RegionOfInterest(5, 1, 15, 10, -1), activation_map)
        ]
        roi = self.detector.filter_rois(rois)
        self.assertEqual(roi,
                         RegionOfInterest(1, 1, 15, 10, -1),
                         "Expected region and actual region have different shape.")
