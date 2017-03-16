import cv2
import picamera
import picamera.array

from sign_detection.model.ImageSource import ImageSource
import sign_detection.tools.image_preprocessing as ip


class PiCameraImageSource(ImageSource):
    def __init__(self, resolution=(300, 200), raw_scale_factor=None, average_value=None):
        # capture from camera at location 0
        self.average_value = average_value
        self.raw_scale_factor = raw_scale_factor
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
        self.camera.capture(output, 'rgb', use_video_port=True)
        img = output.array

        # Flip R and B channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.raw_scale_factor is not None:
            img *= self.raw_scale_factor

        if self.average_value is not None:
            img = ip.set_average_value(img, self.average_value)

        return img
