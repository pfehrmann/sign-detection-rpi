import cv2
import picamera
import picamera.array

from sign_detection.model.ImageSource import ImageSource


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
