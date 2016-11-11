import caffe


def identify_image(model, weights, image_path):
    caffe.set_device(0)
    caffe.set_mode_gpu()

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
