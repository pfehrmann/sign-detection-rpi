from sign_detection.model.IdentifiedImage import IdentifiedImage
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