import caffe
import argparse

import numpy as np
import cv2
from sign_detection.model.PossibleROI import PossibleROI

from sign_detection.model.ScalingSlidingWindow import ScalingSlidingWindow
from sign_detection.model.Sign import get_name_from_category
from sign_detection.tools import lmdb_tools


def initialize_net(model, weights, gpu=True):
    # use either cpu or gpu
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # load and return the net
    return caffe.Net(model, weights, caffe.TEST)


def load_image(image_path, factor=255.0*0.3):
    # load the image and swap channels
    im = caffe.io.load_image(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # return a scaled version of the image. Necessary to reduce the activation of some neurons...
    return im*factor, im


def identify_regions_from_image(model, weights, image_path, gpu=True, minimum=0.99, x=lambda x: 1-x/7, overlap=0.65, factor=255.0*0.3):
    # initialize net and image
    net = initialize_net(model, weights, gpu)
    im, unmodified = load_image(image_path, factor)

    # load image to the net
    rois = identify_regions(net, im, minimum, overlap=overlap, x=x)

    for roi in rois:
        cv2.rectangle(unmodified, (roi.x1, roi.y1), (roi.x2, roi.y2), color=(0, 0, 1), thickness=2)

    # show the image and delay the execution
    cv2.imshow("ROIs", unmodified)
    cv2.waitKey(1000000)

    # save the image. Needs mapping to [0,255]
    cv2.imwrite("result.png", unmodified*255.0)
    return rois


def identify_regions(net, im, minimum=0.99, out_layer='softmax', overlap=0.65, x=lambda x: 1 - 0.2 * x):
    """
    Identify regions in an image.
    :param net: The net
    :param im: The image array
    :param minimum: The minimum certaincy for a ROI to be used
    :param out_layer: The layer yielding the results of the net
    :param overlap: The percentequal overlap between two windows
    :param x: The scaling algorithm for the sliding window
    :return: A list of rois with probabilities
    :returns: list[PossibleROI]
    """

    # return parameters
    rois = []
    number_of_images = 0

    # initialize the sliding window
    window = ScalingSlidingWindow(im, 64, 1, overlap, x)

    # iterate over the scaling sliding window
    for image, roi in window:

        # from time to time output the number of processed windows
        if number_of_images % 1000 == 0:
            print str(number_of_images) + ", " + str(len(rois))
        number_of_images += 1

        # set the data and forward
        net.blobs['data'].data[...] = image
        out = net.forward()

        # get the class
        class_index = out[out_layer].argmax()
        possibility = out[out_layer][0][class_index]

        if possibility >= minimum:
            # create and save the roi
            possible_roi = PossibleROI(roi.x1, roi.y1, roi.x2, roi.y2, roi.sign, possibility)
            rois.append(possible_roi)

            # output information
            class_name = get_name_from_category(class_index)
            print "C: {}, P: {}".format(class_name, possibility)

            # show the image with annotating text
            image = image.transpose(1, 2, 0)
            image = np.uint8(image)
            cv2.putText(img=image, text=class_name, org=(5, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(255, 255, 255), thickness=1)
            cv2.imshow("result", image)

            key = cv2.waitKey(10)
            if key == 27:
                break

    # Show some information about the regions
    print "Images: " + str(number_of_images)
    print "Regions: " + str(rois)
    print "Number Regions: " + str(len(rois))
    return rois


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


def __test(model, weights, gpu=True):
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST)

    for i in range(100, 400):
        image, label = lmdb_tools.get_image_from_lmdb(
            "/home/philipp/PycharmProjects/sign-detection-playground/sign_detection/GTSDB/SlidingWindow/gtsdb_sliding_window",
            "00000" + str(i))
        net.blobs['data'].data[...] = image
        out = net.forward()
        class_index = out['loss'].argmax()

        print "Identifed: " + str(class_index)
        print "Label:     " + str(label)
        cv2.imshow("result", image)


# parse_arguments()
if __name__ == "__main__":
    identify_regions_from_image(
        "C:/Users/phili/PycharmProjects/sign-detection-playground/sign_detection/GTSDB/SlidingWindow/mini_net/deploy.prototxt",
        "C:/Users/phili/PycharmProjects/sign-detection-playground/sign_detection/GTSDB/SlidingWindow/mini_net/snapshot_iter_9210.caffemodel",
        "C:/development/FullIJCNN2013/FullIJCNN2013/00040.ppm", minimum=0.99999, factor=255*0.15, overlap=0.75, x=lambda x: 0.99999-x/6.9)
# test("nin_net_deploy.prototxt", "nin_net/_iter_2000.caffemodel")
