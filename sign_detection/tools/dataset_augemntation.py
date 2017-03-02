import caffe
import cv2
import numpy as np
import sign_detection.GTSRB.ImageReader as ImageReader
import os
import errno
from sign_detection.model.IdentifiedImage import IdentifiedImage
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un

gtsrb_root = "C:/development/GTSRB/Final_Training/Images"
destination_path = "C:/development/GTSRB_AUGMENT/Training/Images"


def blur(image, width, height):
    """
    Blur a given image by the given percentages.
    :type image: np.ndarray
    :type width: float
    :type height: float
    :return: Returns a blurred version of the image
    :returns: np.ndarray
    """
    abs_height = max(int(image.shape[0] * height), 1)
    abs_width = max(int(image.shape[1] * width), 1)

    kernel = np.ones((abs_height, abs_width), np.float32) / (abs_height * abs_width)
    dst = cv2.filter2D(image, -1, kernel)
    return dst


def load_image(image):
    im = caffe.io.load_image(image.path) * 255
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def store_image_class(current_subset, class_index):
    """
    Stores all images from a specific class
    :param current_subset: The list of all images
    :param class_index: The current class
    :return: void
    :type current_subset: list[(sign_detection.model.IdentifiedImage.IdentifiedImage, np.ndarray)]
    :type class_index: int
    """
    assert type(class_index) is int, "class_index is not an integer: %r" % class_index

    prefix = destination_path + '/' + format(class_index, '05d') + '/'  # subdirectory for class
    gt_file = prefix + 'GT-' + format(class_index, '05d') + '.csv'  # annotations file
    create_directory(gt_file)
    csv_text = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n"
    for index, (image, data) in enumerate(current_subset):
        image_name = format(index, '05d') + ".ppm"
        csv_text += image_name + ";" \
                    + str(data.shape[1]) + ";" \
                    + str(data.shape[0]) + ";" \
                    + str(image.get_region_of_interests()[0].x1) + ";" \
                    + str(image.get_region_of_interests()[0].y1) + ";" \
                    + str(image.get_region_of_interests()[0].x2) + ";" \
                    + str(image.get_region_of_interests()[0].y2) + ";" \
                    + str(image.get_region_of_interests()[0].sign) + "\n"
        cv2.imwrite(prefix + image_name, data)

    with open(gt_file, 'w+') as file:
        file.write(csv_text)


def create_directory(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write("FOOBAR")


def store_images(images):
    """
    Stores images under the specified path
    :param images:
    :type images: list[(sign_detection.model.IdentifiedImage.IdentifiedImage, np.ndarray)]
    :return:
    """
    # make sure all images are sorted
    images.sort(key=lambda image: image[0].region_of_interests[0].sign)

    # check each class
    class_index = images[0][0].region_of_interests[0].sign
    current_subset = []
    for image in images:
        if image[0].region_of_interests[0].sign == class_index:
            current_subset.append(image)
        else:
            store_image_class(current_subset, class_index)
            class_index = image[0].region_of_interests[0].sign
            current_subset = [image]
    store_image_class(current_subset, class_index)


def test():
    images = ImageReader.read_train_traffic_signs(gtsrb_root)
    augmented = []  # type: list[(IdentifiedImage, np.ndarray)]
    for index, image in enumerate(images):
        if index % (max(len(images) / 100, 1)) == 0:
            percent_complete = int(index / float(len(images)) * 100)
            print(str(percent_complete) + "% done augmenting...")
        data = load_image(image)
        data = un.set_average_value(data, 100)
        augmented.append((image, data))
        blurred = blur(data, 0.15, 0.01)
        augmented.append((image, blurred))
    store_images(augmented)


if __name__ == "__main__":
    test()
