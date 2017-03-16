import cv2

import sign_detection.EV3.movement as movement
import sign_detection.EV3.images as images
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from sign_detection.GTSDB.multi_processor_detection import RoiResultHandler
import time


class ConsoleHandler(RoiResultHandler):
    """
    :type fps: list[float]
    """

    def __init__(self, num_workers, num_timestamps=1000):
        self.num_timestamps = num_timestamps
        self.num_workers = num_workers
        self.theoretical_fps_list = []
        self.timestamps = [time.time()]
        self.num_image = 0

    def handle_result(self, index, image_timestamp, result_timestamp, rois, possible_rois, image):
        """
        Prints all rois t the console
        :param rois: The rois found
        :return: None
        :type index: int
        :type image_timestamp: float
        :type result_timestamp: float
        :type rois: list[sign_detection.model.PossibleROI.PossibleROI]
        :type possible_rois: list[sign_detection.model.PossibleROI.PossibleROI]
        :type image: numpy.ndarray
        """
        self.timestamps.insert(0, time.time())
        self.timestamps = self.timestamps[:self.num_timestamps]

        real_fps = 1 / ((self.timestamps[0] - self.timestamps[-1]) / len(self.timestamps))

        theoretical_fps = 1.0 / (result_timestamp - image_timestamp)
        self.theoretical_fps_list.insert(0, theoretical_fps)
        self.theoretical_fps_list = self.theoretical_fps_list[:self.num_workers]

        sum_theoretical_fps = 0
        for fps in self.theoretical_fps_list:
            sum_theoretical_fps += fps

        self.num_image += 1

        print str(self.num_image) + ", " + str(time.time()) + ": Real FPS: " + str(real_fps) \
              + ", Theoretical FPS: " + str(sum_theoretical_fps) + " Current FPS: " + \
              str(1.0 / (result_timestamp - image_timestamp))

        for roi in rois:
            print roi.sign


class ViewHandler(RoiResultHandler):
    """
    :type fps: list[float]
    """

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.fps = []

    def handle_result(self, index, image_timestamp, result_timestamp, rois, possible_rois, image):
        """
        Prints all rois t the console
        :param rois: The rois found
        :return: None
        :type index: int
        :type image_timestamp: float
        :type result_timestamp: float
        :type rois: list[sign_detection.model.PossibleROI.PossibleROI]
        :type possible_rois: list[sign_detection.model.PossibleROI.PossibleROI]
        :type image: numpy.ndarray
        """
        current_fps = 1.0 / (result_timestamp - image_timestamp)
        self.fps.insert(0, current_fps)
        self.fps = self.fps[:self.num_workers]

        sum_fps = 0
        for fps in self.fps:
            sum_fps += fps

        un.draw_regions(possible_rois, image, (0, 255, 0))
        un.draw_regions(rois, image, (0, 0, 255), print_class=True)
        cv2.putText(image, "{} fps".format(sum_fps), (5, image.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 1)

        cv2.imshow("Detection", image)
        cv2.waitKey(1)


class EV3Handler(RoiResultHandler):
    """
    :type fps: list[float]
    :type ev3: sign_detection.EV3.EV3.EV3
    """

    def __init__(self, ev3_instance, queue_size=5, min_frame_count=3):
        """

        :param ev3_instance:
        :param queue_size:
        :param min_frame_count:
        :type ev3_instance: sign_detection.EV3.EV3.EV3
        """

        self.min_frame_count = min_frame_count
        self.queue_size = queue_size
        self.ev3 = ev3_instance
        self.last_rois = []

    def handle_result(self, index, image_timestamp, result_timestamp, rois, possible_rois, image):
        """
        Prints all rois t the console
        :param rois: The rois found
        :return: None
        :type index: int
        :type image_timestamp: float
        :type result_timestamp: float
        :type rois: list[sign_detection.model.PossibleROI.PossibleROI]
        :type possible_rois: list[sign_detection.model.PossibleROI.PossibleROI]
        :type image: numpy.ndarray
        """
        self.last_rois.insert(0, rois)
        self.last_rois = self.last_rois[:self.queue_size]

        if is_sign_in_n_rois(1, self.last_rois, self.min_frame_count):
            movement.move(5, 0, self.ev3)
            images.show_image("../prjs/Signs/SIGN_SPEED_30.rgf", self.ev3)

        if is_sign_in_n_rois(13, self.last_rois, self.min_frame_count):
            movement.move(0, 0, self.ev3)
            images.show_image("../prjs/Signs/SIGN_GIVE_WAY.rgf", self.ev3)

        if is_sign_in_n_rois(14, self.last_rois, self.min_frame_count):
            movement.move(0, 0, self.ev3)
            images.show_image("../prjs/Signs/SIGN_STOP.rgf", self.ev3)


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


def is_sign_in_rois(sign, rois):
    """
    :type rois: list[sign_detection.model.RegionOfInterest.RegionOfInterest]
    :param rois: The list of rois to search
    :param sign: The sign to find in the rois
    :return:
    :returns: bool
    """
    for roi in rois:
        if roi.sign == sign:
            return True
    return False
