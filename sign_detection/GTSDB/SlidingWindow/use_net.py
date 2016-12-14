import caffe
import argparse

from scipy.misc.pilutil import imshow

from sign_detection.model.ScalingSlidingWindow import ScalingSlidingWindow
from sign_detection.tools import lmdb_tools


def identify_regions(model, weights, image_path, gpu=True):
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST)

    # load the image in the data layer
    im = caffe.io.load_image(image_path)

    rois = []
    number_of_images = 0

    # initialize the sliding window
    window = ScalingSlidingWindow(im, 64, 1, 0.9, lambda x: 1 - 0.05 * x)
    i = 0
    for image, roi in window:
        if i % 1000 == 0: print str(i) + ", " + str(len(rois))
        net.blobs['data'].data[...] = image
        out = net.forward()
        class_index = out['loss'].argmax()

        if class_index == 1 and out['loss'][0][class_index] > 0.002:
            rois.append(roi)
            print "0: {}, 1: {}".format(out['loss'][0][0], out['loss'][0][1])
            imshow(image)
            # print "found"

        i += 1
        number_of_images += 1

    print "Images: " + str(number_of_images)
    print "Regions: " + str(rois)
    print "Number Regions: " + str(len(rois))


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


def test(model, weights, gpu=True):
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
        imshow(image)

# parse_arguments()
identify_regions("nin_net_deploy.prototxt", "nin_net/_iter_4000.caffemodel",
                 "/home/philipp/development/FullIJCNN2013/00059.ppm")
#test("nin_net_deploy.prototxt", "nin_net/_iter_2000.caffemodel")
