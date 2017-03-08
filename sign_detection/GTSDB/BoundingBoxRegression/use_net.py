import caffe
import cv2

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from sign_detection.model.RegionOfInterest import RegionOfInterest


def use_net():
    model = "net_separate/train.prototxt"
    weights = "data/weights.caffemodel"
    net_bbr = caffe.Net(model, weights, caffe.TEST)

    img = caffe.io.load_image("/home/leifb/Development/Data/GTSDB/00002.ppm")
    detector = load_input_detector()

    data = get_activation(detector, img)

    print 'activation boxes:'
    for roi, activation in data:
        print str(roi)

    corrections = [(bbr(net_bbr, activation), roi) for roi, activation in data]
    corrected_regoins = [RegionOfInterest(v[0][0] + roi.x1, v[0][1] + roi.y1, v[0][2] + roi.x2, v[0][3] + roi.y2, -1)
                         for v, roi in corrections]

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
    out = bbr_net.forward(blobs=['ip3'])
    return out['ip3']


def load_input_detector():
    net = "../ActivationMapBoundingBoxes/mini_net/deploy.prototxt"
    w = "../ActivationMapBoundingBoxes/mini_net/weights.caffemodel"
    net = un.load_net(net, w)
    return un.Detector(net, minimum=0.9999, use_global_max=False, threshold_factor=0.75,
                       draw_results=False,
                       zoom=[1], area_threshold_min=1200, area_threshold_max=50000,
                       activation_layer="conv3",
                       out_layer="softmax", display_activation=False, blur_radius=1, size_factor=0.5,
                       faster_rcnn=True, modify_average_value=True, average_value=30)

use_net()