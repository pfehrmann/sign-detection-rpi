import ImageDataLayerWriter
import ImageReader
import caffe
import random


def train(gtsrb_root_path, max_amount_of_test_images=-1, max_amount_of_train_images=-1):
    """

    :param max_amount_of_test_images: The maximum amount of test images. If -1, all images are used
    :param gtsrb_root_path: eg. "E:/Downloads/GTSRB/Final_Training/Images"
    """
    images = ImageReader.read_traffic_signs(gtsrb_root_path)

    train_images = []
    test_images = []

    train_images += images
    test_images += images

    if (max_amount_of_test_images > 0) and (max_amount_of_test_images < len(test_images)):
        random.seed()
        for i in range(0, len(images) - max_amount_of_test_images):
            # remove random image
            test_index = random.randrange(len(test_images))
            test_images.pop(test_index)

    if (max_amount_of_train_images > 0) and (max_amount_of_train_images < len(train_images)):
        random.seed()
        for i in range(0, len(images) - max_amount_of_train_images):
            # remove random image
            train_index = random.randrange(len(train_images))
            train_images.pop(train_index)

    ImageDataLayerWriter.write_images(train_images, "data/_temp/file_list_train.txt")
    ImageDataLayerWriter.write_images(test_images, "data/_temp/file_list_test.txt")

    # Uncomment to use CPU
    caffe.set_mode_cpu()

    # Uncomment to use GPU
    # caffe.set_device(0)
    # caffe.set_mode_gpu()

    # make sure, that the folder "data/gtsrb" is existing and writable
    solver = caffe.get_solver("lenet_solver.prototxt")
    solver.solve()

    accuracy = 0
    batch_size = solver.test_nets[0].blobs['data'].num
    test_iters = int(len(images) / batch_size)
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
    accuracy /= test_iters

    print("Accuracy: {:.3f}".format(accuracy))


train("E:/Downloads/GTSRB/Final_Training/Images", 6000, 2000)
