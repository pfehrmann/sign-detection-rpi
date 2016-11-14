import csv
from sign_detection.model.IdentifiedImage import IdentifiedImage
from sign_detection.model.RegionOfInterest import RegionOfInterest
import caffe

def read_traffic_signs(root_path):
    """
    Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of identified images
    """
    images = []  # images

    # loop over all 42 classes
    for c in range(0, 43):
        prefix = root_path + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        gtReader.next()  # skip header

        # loop over all images in current annotations file
        for row in gtReader:
            path_to_image = prefix + row[0]

            # find size of image
            roi = [RegionOfInterest(row[3], row[4], row[5], row[6], c)]
            image = IdentifiedImage(path_to_image, roi)
            images.append(image)
        gtFile.close()
    return images
