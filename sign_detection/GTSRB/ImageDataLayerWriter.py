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


def create_files(gtsrb_root_path, max_amount_of_train_images=-1, max_amount_of_test_images=-1):
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

    write_images(train_images, "data/_temp/file_list_train.txt")
    write_images(test_images, "data/_temp/file_list_test.txt")


create_files("C:\Users\phili\Downloads\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images", 10000, 2000)
