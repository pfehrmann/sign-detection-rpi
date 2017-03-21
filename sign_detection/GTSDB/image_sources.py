import time

import cv2
import imutils
import numpy as np

from sign_detection.model.ImageSource import ImageSource


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
    def __init__(self, rotate_angle=0, saturation_factor=3, image_source="E:/2017-03-05 23.10.31.357936/2017-03-05 23.11.32.031385.h264"):
        # capture from camera at location 0
        self.saturation_factor = saturation_factor
        self.rotate_angle = rotate_angle
        self.cap = cv2.VideoCapture(image_source)

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
            time.sleep(1.0 / self.frame_rate)

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
