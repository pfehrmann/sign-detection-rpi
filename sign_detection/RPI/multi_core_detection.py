import sys
caffe_path = "/home/pi/development/caffe/build/install/python"
if not caffe_path in sys.path:
    sys.path.append(caffe_path)

import time
import sign_detection.EV3.EV3 as ev3
import sign_detection.EV3.movement as movement

from sign_detection.GTSDB.multi_processor_detection import Master, RoiResultHandler
from sign_detection.model.ImageSource import ImageSource
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import cv2
import picamera
import picamera.array

num_workers = 5

def create_detector():
    """
    Create a detector
    :return: A Detector
    :returns: sign_detection.GTSDB.Detector_Base
    """
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel")
    un.setup_device(gpu=False)
    detector = un.Detector(net,
                                minimum=0.9999,
                                use_global_max=True,
                                threshold_factor=0.75,
                                draw_results=False,
                                zoom=[1],
                                area_threshold_min=1000,
                                area_threshold_max=500000,
                                activation_layer="activation",
                                out_layer="softmax",
                                display_activation=False,
                                blur_radius=1,
                                size_factor=0.4,
                                max_overlap=0.5,
                                faster_rcnn=True,
                                modify_average_value=False,
                                average_value=60)
    return detector

def test():
    """
    Sets up the master and starts detecting.
    :return:
    """
    global num_workers
    image_source = PiCameraImageSource()
    master = Master(create_detector, image_source, num_workers)
    master.register_roi_result_handler(EV3Handler())
    master.register_roi_result_handler(ConsoleHandler())
    master.start()
    try:
        while 1:
            time.sleep(1)
    except KeyboardInterrupt:
        master.stop()


class PiCameraImageSource(ImageSource):
    def __init__(self, resolution=(300, 200)):
        # capture from camera at location 0
        self.camera = picamera.PiCamera()
        self.camera.resolution = resolution

    def __del__(self):
        self.camera.close()

    def get_next_image(self):
        """
        Returns the nex image from the camera
        :return:
        :returns: numpy.ndarray
        """

        output = picamera.array.PiRGBArray(self.camera)
        self.camera.capture(output, 'rgb', use_video_port=False)
        img = output.array

        # Flip R and B channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class ConsoleHandler(RoiResultHandler):
    """
    :type fps: list[float]
    """
    def __init__(self):
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
        global num_workers
        current_fps = 1.0 / (result_timestamp - image_timestamp)
        self.fps.insert(0, current_fps)
        self.fps = self.fps[:num_workers]

        sum_fps = 0
        for fps in self.fps:
            sum_fps += fps
        print "FPS: " + str(sum_fps)

        for roi in rois:
            print roi.sign


class EV3Handler(RoiResultHandler):
    """
    :type fps: list[float]
    """
    def __init__(self, queue_size=5, min_frame_count=3):
        self.min_frame_count = min_frame_count
        self.queue_size = queue_size
        self.ev3 = ev3.EV3(protocol=ev3.USB, host='00:16:53:47:92:46')
        self.ev3.sync_mode = ev3.ASYNC
        #self.ev3.verbosity = 1
        #movement.move(0, 0, self.ev3)
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

        if is_sign_in_n_rois(14, self.last_rois, self.min_frame_count):
            movement.move(0, 0, self.ev3)


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