from sign_detection.model.IdentifiedImage import IdentifiedImage
import ImageReader
import random
import os
import errno


def write_images(list_of_images, filename):
    """
    :param filename: The name of file to save the generated file to
    :type filename: str
    :type list_of_images: list[IdentifiedImage]
    :return:
    """

    out = ""

    # Construct string
    for image in list_of_images:
        out += image.path + " " + str(image.region_of_interests[0].sign) + "\n"

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write("FOOBAR")

    with open(filename, 'w+') as file:
        file.write(out)


def create_files(gtsrb_train_root_path, gtsrb_test_root_path):
    """

    :param gtsrb_test_root_path: Root of test images
    :param gtsrb_train_root_path: eg. "E:/Downloads/GTSRB/Final_Training/Images"
    """
    train_images = ImageReader.read_train_traffic_signs(gtsrb_train_root_path)
    test_images = ImageReader.read_test_traffic_signs(gtsrb_test_root_path)

    write_images(train_images, "data/_temp/file_list_train.txt")
    write_images(test_images, "data/_temp/file_list_test.txt")

create_files("C:/development/GTSRB/Final_Training/Images", "C:/development/GTSRB/Final_Test/Images")
