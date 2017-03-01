import caffe
import cv2
import numpy as np
import sign_detection.GTSRB.ImageReader as ImageReader

gtsrb_root = "E:/development/GTSRB/Final_Training/Images"
destination_path = ""


def blur(image, width, height):
    """
    Blur a given image by the given percentages.
    :return: Returns a blurred version of the image
    """
    abs_height = max(int(image.shape[0] * height), 1)
    abs_width = max(int(image.shape[1] * width), 1)

    kernel = np.ones((abs_height, abs_width), np.float32)/(abs_height*abs_width)
    dst = cv2.filter2D(image, -1, kernel)
    return dst


def load_image(image):
    im = caffe.io.load_image(image.path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def store_image_class(current_subset, class_index):
    """
    Stores all images from a specific class
    :param current_subset: The list of all images
    :param class_index: The current class
    :return: void
    :type current_subset: list[(sign_detection.model.IdentifiedImage.IdentifiedImage, np.ndarray)]
    """
    pass


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
            current_subset = []


def test():
    images = ImageReader.read_train_traffic_signs(gtsrb_root)
    image = load_image(images[0])
    blurred = blur(image, 0.15, 0.01)
    cv2.imshow("Blurred", blurred)
    cv2.waitKey(10000)

if __name__ == "__main__":
    test()
