import time

from sign_detection.GTSDB.multi_processor_detection import Master, RoiResultHandler
from sign_detection.model.ImageSource import ImageSource
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import cv2


num_workers = 10

def create_detector():
    """
    Create a detector
    :return: A Detector
    :returns: sign_detection.GTSDB.Detector_Base
    """
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net_aug_scale/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/mini_net_aug_scale/weights.caffemodel")
    detector = un.Detector(net,
                                minimum=0.85,
                                use_global_max=False,
                                threshold_factor=0.5,
                                draw_results=False,
                                zoom=[1, 2],
                                area_threshold_min=1000,
                                area_threshold_max=50000,
                                activation_layer="activation",
                                out_layer="softmax",
                                display_activation=False,
                                blur_radius=1,
                                size_factor=0.1,
                                max_overlap=0.2,
                                faster_rcnn=True,
                                modify_average_value=True,
                                average_value=100)
    return detector


def test():
    """
    Sets up the master and starts detecting.
    :return:
    """
    un.setup_device(gpu=True)
    global num_workers
    image_source = CV2ImageSource()
    master = Master(create_detector, image_source, num_workers)
    master.register_roi_result_handler(ConsoleHandler())
    master.start()
    try:
        while 1:
            time.sleep(1)
    except KeyboardInterrupt:
        master.stop()


class ConsoleHandler(RoiResultHandler):
    """
    :type fps: list[float]
    """
    def __init__(self):
        self.fps = []

    def handle_result(self, index, image_timestamp, result_timestamp, rois):
        """
        Prints all rois t the console
        :param rois: The rois found
        :return: None
        :type index: int
        :type image_timestamp: float
        :type result_timestamp: float
        :type rois: list[sign_detection.model.PossibleROI.PossibleROI]
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


class CV2ImageSource(ImageSource):
    def __init__(self):
        # capture from camera at location 0
        self.cap = cv2.VideoCapture(0)

        # Print some of the properties of the camera. For adjustment of speed.
        print "cv2.CAP_PROP_EXPOSURE:   " + str(self.cap.get(cv2.CAP_PROP_EXPOSURE))
        print "cv2.CAP_PROP_APERTURE:   " + str(self.cap.get(cv2.CAP_PROP_APERTURE))
        print "cv2.CAP_PROP_BRIGHTNESS: " + str(self.cap.get(cv2.CAP_PROP_BRIGHTNESS))
        print "cv2.CAP_PROP_CONTRAST:   " + str(self.cap.get(cv2.CAP_PROP_CONTRAST))
        print "cv2.CAP_PROP_SATURATION: " + str(self.cap.get(cv2.CAP_PROP_SATURATION))

        # Change the camera setting using the set() function
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)  # set exposure so we don't have to scale the image
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, True)
        # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 94.0)
        # self.cap.set(cv2.CAP_PROP_SATURATION, 56.0)
        # self.cap.set(cv2.CAP_PROP_CONTRAST, 24.0)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, True)  # set convert to rgb
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_next_image(self):
        """
        Returns the nex image from the camera
        :return:
        :returns: numpy.ndarray
        """
        ret, img = self.cap.read()
        return img

if __name__ == "__main__":
    test()
