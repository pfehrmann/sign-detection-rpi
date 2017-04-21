import caffe
import cv2
import numpy as np

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from sign_detection.model.RegionOfInterest import RegionOfInterest


def use_net():
    model = "net_separate/train.prototxt"
    weights = "data/snapshot/_iter_7200.caffemodel"
    net_bbr = caffe.Net(model, weights, caffe.TEST)

    img = caffe.io.load_image("/home/leifb/Development/Data/GTSDB/00028.ppm")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = load_input_detector()

    data = get_activation(detector, img)

    print 'activation boxes:'
    for roi, activation in data:
        print str(roi)

    corrections = [(bbr(net_bbr, activation), roi) for roi, activation in data]
    corrected_regoins = [bbox_transform_inv(roi, v) for v, roi in corrections]

    print 'bbr corrections:'
    for v, roi in corrections:
        print str(v)

    for d in data:
        un.draw_regions([d[0]], img, color=(1, 0, 0))

    un.draw_regions(corrected_regoins, img, color=(0, 1, 1))

    cv2.imshow('BOXXXES', img)
    cv2.waitKey(3000000)


def get_activation(detector, img):
    regions, activation = detector.identify_regions(img)
    unscaled_regions = [(roi.unscaled, roi) for roi in regions]

    return [(roi, activation[:, :, unscaled.y1:unscaled.y2, unscaled.x1:unscaled.x2])
            for unscaled, roi in unscaled_regions]


def bbr(bbr_net, activation):
    bbr_net.blobs['input_data'].reshape(*activation.shape)
    bbr_net.blobs['input_data'].data[...] = activation
    out = bbr_net.forward(blobs=['ip1'])
    return out['ip1'].copy()


def bbox_transform_inv(roi, delta):
    """
    :param roi:
    :param delta:
    :return:
    :type roi: sign_detection.model.PossibleROI.PossibleROI
    """
    roi = roi.clone()

    width = roi.width + 1.0
    height = roi.height + 1.0
    ctr_x = roi.x1 + 0.5 * width
    ctr_y = roi.y1 + 0.5 * height

    dx = delta[0][0]
    dy = delta[0][1]
    dw = delta[0][2]
    dh = delta[0][3]

    pred_ctr_x = dx * width + ctr_x
    pred_ctr_y = dy * height + ctr_y
    pred_w = np.exp(dw) * width
    pred_h = np.exp(dh) * height

    if np.isinf(pred_w):
        if pred_w > 0:
            pred_w = np.finfo(np.float64).max
        else:
            pred_w = np.finfo(np.float64).min

    if np.isinf(pred_h):
        if pred_h > 0:
            pred_h = np.finfo(np.float64).max
        else:
            pred_h = np.finfo(np.float64).min

    roi.x1 = pred_ctr_x - 0.5 * pred_w
    roi.x2 = pred_ctr_x + 0.5 * pred_w
    roi.y1 = pred_ctr_y - 0.5 * pred_h
    roi.y2 = pred_ctr_y + 0.5 * pred_h

    return roi


def load_input_detector():
    net = "../ActivationMapBoundingBoxes/mini_net/deploy.prototxt"
    w = "../ActivationMapBoundingBoxes/mini_net/weights.caffemodel"
    net = un.load_net(net, w)
    return un.Detector(net,
                       minimum=0.99999,
                       use_global_max=False,
                       threshold_factor=0.75,
                       draw_results=False,
                       zoom=[1],
                       area_threshold_min=400,
                       area_threshold_max=50000,
                       activation_layer="activation",
                       out_layer="softmax",
                       display_activation=False,
                       blur_radius=0,
                       size_factor=0.1,
                       max_overlap=1,
                       faster_rcnn=True,
                       modify_average_value=True,
                       average_value=70)

use_net()
