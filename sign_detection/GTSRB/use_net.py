import caffe
import argparse
import timeit


def load_net(model, weights):
    caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST)

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    return net, transformer

def load_image(image_path, net, transformer):
    # load the image in the data layer
    im = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    return net

def compute(net):
    # compute
    out = net.forward()

    # predicted predicted class
    class_index = out['loss'].argmax()
    return class_index


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Use a trained net to identify images')
    parser.add_argument('image', type=str, nargs='+', help='An image to identify')
    parser.add_argument('-m', '--model', type=str, default='/home/pi/development/nets/quadruple_nin_deploy.prototxt', help='The model to use (.prototxt)')
    parser.add_argument('-w', '--weights', type=str, default='/home/pi/development/nets/quadruple_nin_iter_45000.caffemodel',
                        help='The weights to use (trained net / .caffemodel)')

    # Read the input arguments
    args = parser.parse_args()

    # Only one image allowed for now
    if len(args.image) > 1:
        print 'Only one image allowed for now. Ignoring others.'

    # pass arguments and start identifying
    net, transformer = load_net(args.model, args.weights)
    net = load_image(args.image[0], net, transformer)

    start = timeit.default_timer()

    category = compute(net)

    stop = timeit.default_timer()
    
    print("Time:  " + str(stop - start))
    print "Category: " + str(category)

parse_arguments()
