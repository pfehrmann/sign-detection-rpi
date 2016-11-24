import caffe
from copy import copy
import skimage.io
from sign_detection.model.Window import Window


class SlidingWindow(object):
    def __init__(self, image, width, ratio=1, overlap=0.85):
        """
        Creates a new sliding window. It can be used as an iterator.
        :param image: The image to use (the preprocessed image array).
        :param width: The vertical size of the sliding window in pixels or in percentage,  if <= 1.
        :param ratio: The ratio of the sliding window. If it is 0, the images ratio will be used.
        :param overlap: How much the sliding window will overlap after each step, as fractal.
        """
        """
        Notes:
        In this class, the 'window' is the current box of the sliding window, which is moved though out the iteration,
        while 'image' always refers to the loaded image source.
        """

        # Copy constructor values
        self.image = image
        self.ratio = float(ratio)
        self.overlap = float(overlap)

        # get the image size
        self.image_size = [self.image.shape[2], self.image.shape[1]]

        # initiate window box
        self.window = Window.create(width, self.ratio, self.image_size[0], self.image_size[1])
        self.window.position = [-1, -1]  # An invalid position as a start
        self.step = [int(round(x * (1 - self.overlap))) for x in self.window.size]

    def __move_window(self):
        """
        Moves the window to the next position. If the next position has been reached, a StopIteration will be risen.
        """
        # Check for an 'invalid' position before the window has ever been moved.
        if self.window.x1 < 0 and self.window.y1 < 0:
            self.window.position = [0, 0]
            return

        # Check, if the right side of the image has been reached
        if self.window.reached_right:
            # If the right and bottom have been reached, the last image has been returned.
            if self.window.reached_bottom:
                raise StopIteration()

            # Else, move back to the left and on step down
            new_x = 0
            self.window.reached_right = False

            # Find the next y coordinate and check if it exceeds the image
            new_y = self.window.y1 + self.step[1]
            if new_y + self.window.height >= self.image_size[1]:
                new_y = self.image_size[1] - self.window.height
                self.window.reached_bottom = True

        else:  # Right side has not been reached, move to the right
            new_x = self.window.x1 + self.step[0]
            new_y = self.window.y1
            # If the new x exceeds the image
            if new_x + self.window.width >= self.image_size[0]:
                new_x = self.image_size[0] - self.window.width
                self.window.reached_right = True

        # move the window
        self.window.position = [new_x, new_y]

    def __iter__(self):
        """
        Note: Python wants a object that can be used in a for loop to return a iterator when calling the __iter__().
        Since the SlidingWindow is the iterator of itself, just return self.
        """
        return self

    def next(self):
        """
        Moves the window to the next position and returns the corresponding excerpt. The region of interest marks the
        position of the excerpt within its source image.
        :return An excerpt of the source image and a region of interest.
        :returns: numpy.ndarray, RegionOfInterest
        """
        self.__move_window()
        return_image = self.image[
                       0:3,
                       self.window.y1: self.window.y2,
                       self.window.x1: self.window.x2]
        return copy(return_image), copy(self.window)


def test():

    i = caffe.io.load_image('/home/leifb/Downloads/Schilder/Vorfahrt.png')
    model = '../GTSRB/quadruple_nin_deploy.prototxt'
    weights = '../GTSRB/data/gtsrb/quadruple_nin_iter_20000.caffemodel'

    # load input and configure prepossessing
    transformer = caffe.io.Transformer({'data': (1, 3, 412, 963)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    processed = transformer.preprocess('data', i)

    s = SlidingWindow(processed, 300, 0, 0.5)

    transformer2 = caffe.io.Transformer({'data': (1, 3, 412, 963)})
    # transformer2.set_channel_swap('data', (2, 1, 0))
    transformer2.set_raw_scale('data', 1.0/255.0)

    it = 0
    for image, roi in s:
        # back = transformer2.preprocess('data', image)
        transposed = image.transpose(1, 2, 0)
        scaled = transformer2.preprocess('data', transposed)
        skimage.io.imsave('image/%s.bmp' % it, scaled)
        it += 1

# test()
