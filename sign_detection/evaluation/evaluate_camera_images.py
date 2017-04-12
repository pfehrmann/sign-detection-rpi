import time

import cv2

import sign_detection.GTSDB.multi_processor_detection as mpd
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from sign_detection.GTSDB.image_sources import CV2CameraImageSource, CV2VideoImageSource
from sign_detection.GTSDB.roi_result_handlers import ViewHandler, ConsoleHandler
from sign_detection.tools.image_preprocessing import set_average_value

average_value = 35

net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
                  "../GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel")
un.setup_device(gpu=True)
detector = un.Detector(net,
                       minimum=0.999,
                       use_global_max=False,
                       threshold_factor=0.75,
                       draw_results=False,
                       zoom=[1, 2],
                       area_threshold_min=1000,
                       area_threshold_max=4900,
                       activation_layer="activation",
                       out_layer="softmax",
                       global_pooling_layer="pool1",
                       display_activation=False,
                       blur_radius=1,
                       size_factor=0.0,
                       max_overlap=0.5,
                       faster_rcnn=True,
                       modify_average_value=True,
                       average_value=average_value)


def getImages(image_path):
    scaled_raw, image_raw = un.load_image(image_path)

    img = image_raw * 255.0

    rois, unfiltered = detector.identify_regions_from_image(img[:])

    draw = image_raw[:] * 255.0
    un.draw_regions(unfiltered, draw, (0, 255, 0))
    un.draw_regions(rois, draw, (0, 0, 255), print_class=True)

    cv2.imshow("Regions", draw / 255.0)

    average_image = set_average_value(img[:], average_value)

    cv2.imshow("ModifiedAverage", average_image)
    # cv2.waitKey(10000000)

    return average_image, draw


if __name__ == "__main__":
    images = ["C:/Users/phili/Dropbox/Uni/Studienarbeit/Arbeit/Images/VideoSnaps/1.png",
              "C:/Users/phili/Dropbox/Uni/Studienarbeit/Arbeit/Images/VideoSnaps/2.png",
              "C:/Users/phili/Dropbox/Uni/Studienarbeit/Arbeit/Images/VideoSnaps/4.png",
              "C:/Users/phili/Dropbox/Uni/Studienarbeit/Arbeit/Images/VideoSnaps/3.png"]

    for image_path in images:
        average_image, regions = getImages(image_path)
        cv2.imwrite(image_path + ".average.png", img=average_image)
        cv2.imwrite(image_path + ".regions.png", img=regions)
