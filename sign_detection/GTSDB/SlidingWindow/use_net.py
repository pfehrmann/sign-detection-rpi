import caffe
import argparse

from sign_detection.model.ScalingSlidingWindow import ScalingSlidingWindow


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
    window = ScalingSlidingWindow(im, 64, 1, 0.85, lambda x: 1/(x+1))
    for image, roi in window:
        net.blobs['data'].data[...] = image
        out = net.forward()
        class_index = out['loss'].argmax()

        if class_index == 1:
            rois.append(roi)

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

# parse_arguments()
identify_regions("model.prototxt", "model.caffemodel", "E:/Downloads/TrainIJCNN2013/TrainIJCNN2013/00012.ppm")
