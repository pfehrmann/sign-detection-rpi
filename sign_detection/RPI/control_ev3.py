import sys
caffe_path = "/home/pi/development/caffe/build/install/python"
if not caffe_path in sys.path:
    sys.path.append(caffe_path)

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import cv2
from time import time
import picamera
import picamera.array
import sign_detection.EV3.movement as movement
import sign_detection.EV3.EV3 as ev3


resultion=(300, 200)
queue_size = 4
count_rois = 3

def test():
    global resultion
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel")
    detector = un.Detector(net,
                           minimum=0.999,
                           use_global_max=False,
                           threshold_factor=0.75,
                           draw_results=False,
                           zoom=[1],
                           area_threshold_min=1000,
                           area_threshold_max=50000,
                           activation_layer="activation",
                           out_layer="softmax",
                           display_activation=False,
                           blur_radius=1,
                           size_factor=0.5,
                           faster_rcnn=True,
                           modify_average_value=False,
                           average_value=100)

    # rois
    last_rois = []
    with picamera.PiCamera() as camera:
        camera.resolution = resultion
        my_e_v3 = ev3.EV3(protocol=ev3.USB, host='00:16:53:47:92:46')
        try:
            while True:
                start = time()

                # capture the image
                output = picamera.array.PiRGBArray(camera)
                camera.capture(output, 'rgb', use_video_port=True)
                img = output.array

                # Flip R and B channel
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # pass the image through the net
                rois, unfiltered = detector.identify_regions_from_image(img, img)

                last_rois.insert(0, rois)
                if len(last_rois) > queue_size:
                    last_rois = last_rois[:queue_size]

                if is_sign_in_n_rois(1, last_rois, count_rois):
                    movement.move(30, 0, my_e_v3)

                if is_sign_in_n_rois(14, last_rois, count_rois):
                    movement.move(0, 0, my_e_v3)

                end = time()

                fps = "{} fps".format(1.0 / (end - start))
                print fps
        finally:
            # clean up
            cv2.destroyAllWindows()
            cv2.VideoCapture(0).release()


def is_sign_in_n_rois(sign, list_rois, n):
    """
    :type list_rois: list[list[sign_detection.model.RegionOfInterest.RegionOfInterest]]
    :type n: int
    :param sign:
    :param list_rois:
    :return:
    :returns: bool
    """
    count = 0
    for rois in list_rois:
        if is_sign_in_rois(sign, rois):
            count += 1
    return count > n


def is_sign_in_all_rois(sign, list_rois):
    """
    :type list_rois: list[list[sign_detection.model.RegionOfInterest.RegionOfInterest]]
    :param sign:
    :param list_rois:
    :return:
    :returns: bool
    """
    for rois in list_rois:
        if not is_sign_in_rois(sign, rois):
            return False
    return True

def is_sign_in_rois(sign, rois):
    """
    :type rois: list[sign_detection.model.RegionOfInterest.RegionOfInterest]
    :param rois:
    :return:
    :returns: bool
    """
    for roi in rois:
        if roi.sign == sign:
            return True
    return False

if __name__ == "__main__":
    test()