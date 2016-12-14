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


def supply_image(image_array, net, transformer):
    net.blobs['data'].data[...] = transformer.preprocess('data', image_array)
    return net

def load_image(image_path, net, transformer):
    # load the image in the data layer
    im = caffe.io.load_image(image_path)
    return supply_image(im, net, transformer)


def compute(net):
    # compute
    out = net.forward()

    # predicted predicted class
    class_index = out['loss'].argmax()
    return class_index, out['loss'][0][class_index]


def get_name_from_category(category):
    category = int(category)
    categories = {
        0: "speed limit 20",
        1: "speed limit 30",
        2: "speed limit 50",
        3: "speed limit 60",
        4: "speed limit 70",
        5: "speed limit 80",
        6: "restriction ends 80",
        7: "speed limit 100",
        8: "speed limit 120",
        9: "no overtaking",
        10: "no overtaking (trucks)",
        11: "priority at next intersection ",
        12: "priority road",
        13: "give way",
        14: "stop",
        15: "no traffic both ways",
        16: "no trucks",
        17: "no entry",
        18: "danger",
        19: "bend left",
        20: "bend right",
        21: "bend",
        22: "uneven road",
        23: "slippery road",
        24: "road narrows",
        25: "construction",
        26: "traffic signal",
        27: "pedestrian crossing",
        28: "school crossing",
        29: "cycles crossing",
        30: "snow",
        31: "animals",
        32: "restriction ends",
        33: "go right",
        34: "go left",
        35: "go straight",
        36: "go right or straight",
        37: "go left or straight",
        38: "keep right",
        39: "keep left",
        40: "roundabout",
        41: "restriction ends (overtaking)",
        42: "restriction ends (overtaking (trucks))",
        43: "no sign"}
    return categories[category]

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Use a trained net to identify images')
    parser.add_argument('image', type=str, nargs='+', help='An image to identify')
    parser.add_argument('-m', '--model', type=str, default='/home/pi/development/nets/quadruple_nin_deploy.prototxt',
                        help='The model to use (.prototxt)')
    parser.add_argument('-w', '--weights', type=str,
                        default='/home/pi/development/nets/quadruple_nin_iter_45000.caffemodel',
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

    category, probability = compute(net)

    stop = timeit.default_timer()

    print("Time:  " + str(stop - start))
    print "Category: " + str(category) + ": " + get_name_from_category(category) + ", " + str(probability)


#parse_arguments()
