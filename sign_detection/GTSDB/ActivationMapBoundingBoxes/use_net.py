import caffe
import argparse

import numpy as np
import cv2
from sign_detection.model.RegionOfInterest import RegionOfInterest
from sign_detection.model.PossibleROI import PossibleROI
from time import time
from matplotlib import pyplot as plt
from sign_detection.model.Sign import get_name_from_category


def initialize_net(model, weights, gpu=True):
    # use either cpu or gpu
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # load and return the net
    return caffe.Net(model, weights, caffe.TEST)


def load_image(image_path, factor=255.0 * 0.3):
    # load the image and swap channels
    im = caffe.io.load_image(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # return a scaled version of the image. Necessary to reduce the activation of some neurons...
    return im * factor, im


def identify_regions_from_image(model, weights, image_path, gpu=True, minimum=0.99, factor=255.0 * 0.3,
                                use_global_max=True, threshold_factor=0.5, draw_results=False, zoom=[1, 2, 3],
                                area_thrshold_min=49, area_thrshold_max=10000, activation_layer="conv3",
                                out_layer="softmax", display_activation=False, blur_radius=1):
    """
    Load and process a net and image
    :param model: The path to the prototxt model definition
    :param weights: The path to the caffemodel weights file
    :param image_path: The path to the image
    :param gpu: Use the GPU? Default is true
    :param minimum: The minimum probability of a class. Everything below is discarded
    produced. Should be in range [0, 1).
    :param factor: The facor to multiply the image with. Use this to prevent over stimulation.
    :param use_global_max: Use the global maximum or use local maxima of filters?
    :param threshold_factor: The threshold is defined by the maximum * threshold_factor
    :param draw_results: This outputs the found results visibly
    :return: Returns all the found ROIs as a list of PossibleROI elements.
    :returns: list[PossibleROI]
    """

    # initialize net and image
    net = initialize_net(model, weights, gpu)
    im, unmodified = load_image(image_path, factor)

    start = time()

    # collect all the regions of interest
    overlapping_rois = []
    for step in zoom:
        factor = 1.0 / step
        resized = cv2.resize(im, None, fx=factor, fy=factor)
        new_regions = identify_regions(net, resized, use_global_max=use_global_max,
                                       threshold_factor=threshold_factor,
                                       draw_results=draw_results, area_threshold_min=area_thrshold_min,
                                       area_threshold_max=area_thrshold_max,
                                       activation_layer=activation_layer,
                                       display_activation=display_activation,
                                       blur_radius=blur_radius)

        for roi in new_regions:
            roi.x1 *= step
            roi.x2 *= step
            roi.y1 *= step
            roi.y2 *= step

        overlapping_rois.extend(new_regions)

    # remove overlapping regions
    # unfiltered_rois = filter_rois(overlapping_rois, max_overlap=0.75)
    unfiltered_rois = overlapping_rois
    print "Checking {} rois".format(len(unfiltered_rois))

    # check each roi individually
    __check_rois(im, net, net.blobs['data'].shape, out_layer, unfiltered_rois)

    # filter all the rois with a too low possibility
    rois = [roi for roi in unfiltered_rois if roi.probability >= minimum]

    if True or draw_results:
        for roi in unfiltered_rois:
            cv2.rectangle(unmodified, (int(roi.x1), int(roi.y1)), (int(roi.x2), int(roi.y2)), color=(0, 1, 0),
                          thickness=2)

    # draw the regions
    for roi in rois:
        cv2.rectangle(unmodified, (int(roi.x1), int(roi.y1)), (int(roi.x2), int(roi.y2)), color=(0, 0, 1), thickness=2)

    end = time()
    print "Total time: " + str(end - start)

    # show the image and delay the execution
    cv2.imshow("ROIs", unmodified)
    cv2.waitKey(1000000)

    # save the image. Needs mapping to [0,255]
    cv2.imwrite("result.png", unmodified * 255.0)
    return rois


def filter_rois(rois, max_overlap):
    all_regions = rois[:]
    result = []
    for roi in rois:
        keep = True
        for other in all_regions:
            if roi is not other and roi.area <= other.area and roi.get_overlap(other) > max_overlap:
                keep = False
                break
        if keep:
            result.append(roi)
        else:
            all_regions.remove(roi)
    return result


def identify_regions(net, image, out_layer='softmax', activation_layer="conv3", use_global_max=True,
                     threshold_factor=0.5, draw_results=False, area_threshold_min=49, area_threshold_max=10000,
                     display_activation=False, blur_radius=1):
    """
    Identify regions in an image.
    :param display_activation: If True all activation maps will be displayed
    :param area_threshold_min: The minimum size of an area
    :param net: The net
    :param image: The image array
    :param out_layer: The layer yielding the results of the net
    :param activation_layer: The layer that yields the activation map
    :param use_global_max: Use the global maximum or use local maxima of filters?
    :param threshold_factor: The threshold is defined by the maximum * threshold_factor
    :param draw_results: This outputs the found results visibly
    :return: A list of rois with probabilities
    :returns: list[PossibleROI]
    :type net: caffe.Net
    """

    # return parameters
    rois = []

    # Transpose to fit caffes needs
    caffe_in = image.transpose((2, 0, 1))

    # store the original shape of the input layer
    original_shape = net.blobs['data'].shape

    # reshape the input layer to match the images size
    width = caffe_in.shape[1]
    height = caffe_in.shape[2]
    net.blobs['data'].reshape(1, 3, width, height)

    # set the data and forward
    net.blobs['data'].data[...] = caffe_in
    out = net.forward(blobs=[out_layer, activation_layer])

    # get the activation for the proposals from the activation layer
    activation = out[activation_layer]
    global_max = activation.max()
    factor_y = image.shape[0] / activation.shape[2]
    factor_x = image.shape[1] / activation.shape[3]

    if display_activation:
        display_activation_maps(activation)

    for filter_index in range(len(activation[0])):

        # analyze image
        filter = activation[0][filter_index]
        regions, contours = __get_regions_from_filter(factor_x, factor_y, filter, global_max, threshold_factor,
                                                      use_global_max, area_threshold_min=area_threshold_min,
                                                      area_threshold_max=area_threshold_max, blur_radius=blur_radius)

        rois.extend(regions)

        if draw_results:
            draw_contours(filter, contours)

    # reset the shape
    net.blobs['data'].reshape(original_shape[0], original_shape[1], original_shape[2], original_shape[3])

    # Show some information about the regions
    print "Number Regions: " + str(len(rois))
    return rois


def __draw_regions(rois, image):
    for roi in rois:
        cv2.rectangle(image, (int(roi.x1), int(roi.y1)), (int(roi.x2), int(roi.y2)), color=(0, 0, 1), thickness=2)


def display_activation_maps(layer_blob):
    plot = 1
    count_plots = layer_blob.shape[1]
    width = int(count_plots ** 0.5)
    height = width + 1
    for map in layer_blob[0]:
        plt.subplot(width, height, plot), plt.imshow(map, 'gray')
        plot += 1
    plt.show()


def __get_regions_from_filter(factor_x, factor_y, filter, global_max, threshold_factor,
                              use_global_max, area_threshold_min=49, area_threshold_max=10000, blur_radius=1):
    rois = []
    max_value = filter.max()
    if use_global_max:
        threshold_value = global_max * threshold_factor
    else:
        threshold_value = max_value * threshold_factor

    # apply threshold
    ret, thresh = cv2.threshold(filter, threshold_value, 0, 3)

    # blur the image to produce better results
    if blur_radius > 1:
        thresh = cv2.blur(thresh, (blur_radius, blur_radius))

    # extract the contours
    converted = np.array(thresh / max_value * 255, dtype=np.uint8)
    im2, contours, hierarchy = cv2.findContours(converted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create bounding boxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        h = max(abs(w), abs(h))
        w = max(abs(w), abs(h))
        area = w * h * factor_x * factor_y
        if area >= area_threshold_min and area <= area_threshold_max:
            # append the found roi to the list of rois
            rois.append(PossibleROI(x * factor_x, y * factor_y, (x + w) * factor_x, (y + h) * factor_y, -1, 0))
    return rois, contours


def __check_rois(image, net, original_shape, out_layer, rois):
    for roi in rois:
        copy = RegionOfInterest(roi.x1, roi.y1, roi.x2, roi.y2, roi.sign)
        copy.increase_size(0.4)
        copy.x1 = max(0, copy.x1)
        copy.y1 = max(0, copy.y1)
        copy.x2 = min(image.shape[1], copy.x2)
        copy.y2 = min(image.shape[0], copy.y2)

        crop_img = image[copy.y1:copy.y2, copy.x1:copy.x2]
        crop_img = caffe.io.resize_image(crop_img, (original_shape[2], original_shape[3]))
        caffe_in = crop_img.transpose((2, 0, 1))
        net.blobs['data'].data[...] = caffe_in
        out = net.forward()

        # get the class
        class_index = out[out_layer].argmax()
        possibility = out[out_layer][0][class_index]

        roi.probability = possibility
        roi.sign = class_index


def draw_contours(image, contours):
    # draw the results
    draw = np.array(image / image.max() * 255, dtype=np.uint8)
    draw = cv2.cvtColor(draw, cv2.COLOR_GRAY2RGB)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        h = max(w, h)
        w = max(w, h)
        area = w * h
        if area >= 49:
            cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Regions", draw)
    cv2.waitKey(1000)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Use a trained net to identify images')
    parser.add_argument('image', type=str, nargs='+', help='An image to identify')
    parser.add_argument('-m', '--model', type=str, default='model.prototxt', help='The model to use (.prototxt)')
    parser.add_argument('-w', '--weights', type=str, default='weights.caffemodel',
                        help='The weights to use (trained net / .caffemodel)')
    parser.add_argument('-g', '--gpu', type=bool, default=True, help='Use the GPU to solve?')

    # Read the input arguments
    args = parser.parse_args()

    # Only one image allowed for now
    if len(args.image) > 1:
        print 'Only one image allowed for now. Ignoring others.'

    # pass arguments and start identifying
    identify_regions(args.model, args.weights, args.image[0], args.gpu)


# parse_arguments()
if __name__ == "__main__":
    regions = identify_regions_from_image(
        "C:/Users/phili/PycharmProjects/sign-detection-playground/sign_detection/GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
        "C:/Users/phili/PycharmProjects/sign-detection-playground/sign_detection/GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel",
        "C:/development/FullIJCNN2013/FullIJCNN2013/00000.ppm", minimum=0.999, factor=255 * 0.15, use_global_max=True,
        threshold_factor=0.5, draw_results=False, zoom=[1, 3, 6], area_thrshold_min=600, area_thrshold_max=50000,
        activation_layer="activation",
        display_activation=False, gpu=True, blur_radius=1)

    for roi in regions:
        print get_name_from_category(roi.sign) + " (" + str(roi.probability) + ") @({},{}), ({},{})".format(roi.x1,
                                                                                                            roi.y1,
                                                                                                            roi.x2,
                                                                                                            roi.y2)
