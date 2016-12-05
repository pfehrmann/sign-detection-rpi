import caffe
import argparse
import numpy as np

from sign_detection.model.ScalingSlidingWindow import ScalingSlidingWindow


def identify_image(model, weights, image_path):
    # caffe.set_device(0)
    caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST)

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    # load the image in the data layer
    im = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    # compute
    out = net.forward()

    # predicted predicted class
    class_index = out['loss'].argmax()
    return class_index

def identify_regions(model, weights, image_path):
    # caffe.set_device(0)
    caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST)

    # load the image in the data layer
    im = caffe.io.load_image(image_path)
    processed = preprocess(im)

    rois = []
    i = 0
    window = ScalingSlidingWindow(processed, 64, 1, 0.85, lambda x: 1/(x+2))
    for image, roi in window:
        net.blobs['data'].data[...] = image
        out = net.forward()
        class_index = out['loss'].argmax()
        i += 1
        if class_index == 2:
            rois.append(roi)
        if i%200==0: print i
    print i
    print(rois)
    print len(rois)
    return class_index


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Use a trained net to identify images')
    parser.add_argument('image', type=str, nargs='+', help='An image to identify')
    parser.add_argument('-m', '--model', type=str, default='model.prototxt', help='The model to use (.prototxt)')
    parser.add_argument('-w', '--weights', type=str, default='weights.caffemodel',
                        help='The weights to use (trained net / .caffemodel)')

    # Read the input arguments
    args = parser.parse_args()

    # Only one image allowed for now
    if len(args.image) > 1:
        print 'Only one image allowed for now. Ignoring others.'

    # pass arguments and start identifying
    identify_image(args.model, args.weights, args.image[0])
    identify_regions(args.model, args.weights, args.image[0])

def preprocess(im):
    """
    preprocess() emulate the pre-processing occuring in the vgg16 caffe
    prototxt.
    """

    im = np.float32(im)
    #im = im[:, :, ::-1]  # change to BGR
    #im = im.transpose((2, 0, 1))

    return im

#parse_arguments()
print identify_regions("model.prototxt", "model.caffemodel", "C:/Users/phili/Downloads/FullIJCNN2013/FullIJCNN2013/00002.ppm")