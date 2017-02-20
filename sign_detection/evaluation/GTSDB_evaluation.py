import cv2

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import sign_detection.GTSDB.SlidingWindow.batchloader as BatchLoader
from time import time
import caffe.io


def load(image):
    return caffe.io.load_image(image.path)


def test(gpu=True):
    # initialize caffe
    un.setup_device(gpu=gpu)

    # Setup the net and transformer
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel")

    # setup the detector
    detector = un.Detector(net, minimum=0.9999, use_global_max=False, threshold_factor=0.75, draw_results=False,
                           zoom=[1], area_threshold_min=2000, area_threshold_max=30000, activation_layer="activation",
                           out_layer="softmax", display_activation=False, blur_radius=2, size_factor=0.4,
                           max_overlap=0.01)

    images = BatchLoader.get_images_and_regions(gtsdb_root="C:\development\FullIJCNN2013\FullIJCNN2013", min=600,
                                                max=899)

    for image in images:
        image_raw = load(image)
        rois, unfiltered = detector.identify_regions_from_image(image_raw, image_raw)
        evaluate(rois, image.get_region_of_interests)

    # clean up
    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()


def evaluate(found_rois, expected_rois):
    correct_rois = []
    for expected_roi in expected_rois:
        for found_roi in found_rois:
            if found_roi.similar(expected_roi, 0.25):
                correct_rois.append((found_roi, expected_roi))

    false_negative = expected_rois[:]
    for expected_roi in expected_rois:
        for (found, expected) in correct_rois:
            if expected == expected_roi:
                false_negative.remove(expected)

    false_positive = found_rois[:]
    for found_roi in found_rois:
        for (found, expected) in correct_rois:
            if found == found_roi:
                false_positive.remove(found)

    return correct_rois, false_negative, false_positive


if __name__ == '__main__':
    test(gpu=True)
