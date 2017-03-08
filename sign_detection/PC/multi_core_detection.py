import time

from sign_detection.GTSDB.multi_processor_detection import Master, RoiResultHandler
from sign_detection.model.ImageSource import ImageSource
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
import cv2
import imutils
import numpy as np


num_workers = 4

def create_detector():
    """
    Create a detector
    :return: A Detector
    :returns: sign_detection.GTSDB.Detector_Base
    """
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel")
    un.setup_device(gpu=True)
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
    image_source = CV2VideoImageSourceRealTime(173)
    master = Master(create_detector, image_source, num_workers)
    master.register_roi_result_handler(ViewHandler())
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


class ViewHandler(RoiResultHandler):
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

        un.draw_regions(possible_rois, image, (0, 255, 0))
        un.draw_regions(rois, image, (0, 0, 255), print_class=True)
        cv2.putText(image, "{} fps".format(sum_fps), (5, image.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 1)

        cv2.imshow("Detection", image)
        cv2.waitKey(1)


class CV2CameraImageSource(ImageSource):
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


class CV2VideoImageSource(ImageSource):
    def __init__(self, rotate_angle=0, saturation_factor=3):
        # capture from camera at location 0
        self.saturation_factor = saturation_factor
        self.rotate_angle = rotate_angle
        self.cap = cv2.VideoCapture("E:/2017-03-05 23.10.31.357936/2017-03-05 23.11.32.031385.h264")

    def get_next_image(self):
        """
        Returns the nex image from the camera
        :return:
        :returns: numpy.ndarray
        """
        ret, img = self.cap.read()

        if self.rotate_angle != 0:
            img = imutils.rotate(img, self.rotate_angle)

        imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
        (h, s, v) = cv2.split(imghsv)
        s *= self.saturation_factor
        s = np.clip(s, 0, 255)
        imghsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(imghsv, cv2.COLOR_HSV2RGB_FULL)
        return img


class CV2VideoImageSourceRealTime(ImageSource):
    def __init__(self, rotate_angle=0, saturation_factor=1, frame_rate=25.0):
        # capture from camera at location 0
        self.frame_rate = frame_rate
        self.saturation_factor = saturation_factor
        self.rotate_angle = rotate_angle
        self.cap = cv2.VideoCapture("E:/2017-03-05 23.10.31.357936/2017-03-05 23.13.32.099571.h264")
        self.image = None
        import thread
        thread.start_new_thread(self._worker, ())

    def _worker(self):
        go_on = True
        while go_on:
            go_on, self.image = self.cap.read()
            time.sleep(1.0/self.frame_rate)

    def get_next_image(self):
        """
        Returns the nex image from the camera
        :return:
        :returns: numpy.ndarray
        """
        img = self.image[:]
        if self.rotate_angle != 0:
            img = imutils.rotate(img, self.rotate_angle)

        if self.saturation_factor != 1:
            imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
            (h, s, v) = cv2.split(imghsv)
            s = s * self.saturation_factor
            s = np.clip(s, 0, 255)
            imghsv = cv2.merge([h, s, v])
            img = cv2.cvtColor(imghsv, cv2.COLOR_HSV2RGB_FULL)


        return img



if __name__ == "__main__":
    test()
