import math

import caffe.io
import cv2
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import sign_detection.GTSDB.SlidingWindow.batchloader as BatchLoader


def load(image):
    im = caffe.io.load_image(image.path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def test(gpu=True):
    # initialize caffe
    un.setup_device(gpu=gpu)

    # Setup the net and transformer
    path = "mini_net_aug_scale"
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/" + path + "/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/" + path + "/weights.caffemodel")

    # setup the detector
    detector = un.Detector(net,
                           minimum=0.9,
                           use_global_max=False,
                           threshold_factor=0.75,
                           draw_results=False,
                           zoom=[1, 2],
                           area_threshold_min=1000,
                           area_threshold_max=50000,
                           activation_layer="activation",
                           out_layer="softmax",
                           display_activation=False,
                           blur_radius=1,
                           size_factor=0.1,
                           max_overlap=0.5,
                           faster_rcnn=True,
                           modify_average_value=True,
                           average_value=95)

    images = BatchLoader.get_images_and_regions(gtsdb_root="C:/development/FullIJCNN2013/FullIJCNN2013", min=0,
                                                max=900, shuffle_result=False)
    images = images[600:620]
    correct_rois = []
    false_negatives = []
    false_positives = []
    for image in images:
        image_raw = load(image) * 255
        rois, unfiltered = detector.identify_regions_from_image(image_raw, image_raw)
        correct, false_negative, false_positive = evaluate(rois, image.get_region_of_interests())
        correct_rois.extend(correct)
        false_negatives.extend(false_negative)
        false_positives.extend(false_positive)

    # create the two vectors for the prediction
    true_labels = []
    scores = []
    for found, expected in correct_rois:
        true_labels.append(expected.sign)
        scores.append(expected.sign)

    for false_negative in false_negatives:
        true_labels.append(false_negative.sign)
        scores.append(-1)

    for false_positive in false_positives:
        true_labels.append(-1)
        scores.append(false_positive.sign)

    true_labels = [int(val) for val in true_labels]
    scores = [int(val) for val in scores]

    true_labels = label_binarize(true_labels, classes=range(-1, 43))
    scores = label_binarize(scores, classes=range(-1, 43))

    average_precision = [0 for x in range(44)]
    for i in range(44):
        average_precision[i] = average_precision_score(true_labels[:, i], scores[:, i], average='macro')

    average_precision_micro = average_precision_score(true_labels, scores, average="micro")

    # Remove the nan values. This enables us to get a number from the mAP macro. But it seems like this numbeis not
    # quite right...
    cleaned_ap = [x for x in average_precision if not math.isnan(x)]

    # see http://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification for
    # specification od modes. It seems, that micro is the mode used in PASCAL VOC
    print "AP (micro):  " + str(average_precision_micro)
    print "mAP (macro): " + str(np.mean(average_precision))
    print "cleaned mAP: " + str(np.mean(cleaned_ap))
    print "True Positives:  " + str(len(correct_rois))
    print "False Positives: " + str(len(false_positives))
    print "False Negatives: " + str(len(false_negatives))

    store_results("C:\Users\phili\Dropbox\Uni\Studienarbeit\GTSDB_Results.csv", detector, average_precision_micro,
                  np.mean(average_precision), np.mean(cleaned_ap),
                  len(correct_rois), len(false_positives), len(false_negatives), len(images), path)


def store_results(file, detector, ap_micro, m_ap, m_ap_cleaned, true_positives, false_positives, false_negatives,
                  num_images, path):
    result = str(ap_micro) + ";" \
             + str(m_ap) + ";" \
             + str(m_ap_cleaned) + ";" \
             + str(true_positives) + ";" \
             + str(false_positives) + ";" \
             + str(false_negatives) + ";" \
             + str(num_images) + ";" \
             + str(detector.minimum) + ";" \
             + str(detector.use_global_max) + ";" \
             + str(detector.threshold_factor) + ";" \
             + str(detector.draw_results) + ";" \
             + str(detector.zoom) + ";" \
             + str(detector.area_threshold_min) + ";" \
             + str(detector.area_threshold_max) + ";" \
             + str(detector.activation_layer) + ";" \
             + str(detector.out_layer) + ";" \
             + str(detector.display_activation) + ";" \
             + str(detector.blur_radius) + ";" \
             + str(detector.size_factor) + ";" \
             + str(detector.max_overlap) + ";" \
             + str(detector.faster_rcnn) + ";" \
             + str(detector.modify_average_value) + ";" \
             + str(detector.average_value) + ";" \
             + str(path)
    result += "\n"
    print "Saving results..."
    print(result)
    fd = open(file, 'a')
    fd.write(result)
    fd.close()

def evaluate(found_rois, expected_rois):
    # Find the true positives
    correct_rois = []
    for expected_roi in expected_rois:
        for found_roi in found_rois:
            if found_roi.similar(expected_roi, 0.0):
                correct_rois.append((found_roi, expected_roi))

    # Find all the false negatives
    false_negative = expected_rois[:]
    for expected_roi in expected_rois:
        for (found, expected) in correct_rois:
            if expected == expected_roi and expected in false_negative:
                false_negative.remove(expected)

    # Find all the false positives
    false_positive = found_rois[:]
    for found_roi in found_rois:
        for (found, expected) in correct_rois:
            if found == found_roi and found in false_positive:
                false_positive.remove(found)

    return correct_rois, false_negative, false_positive


if __name__ == '__main__':
    test(gpu=True)
