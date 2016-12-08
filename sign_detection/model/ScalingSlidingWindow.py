import caffe.io
from skimage.io import imsave
import warnings

from sign_detection.model.SlidingWindow import SlidingWindow
from sign_detection.model.Window import Window


class ScalingSlidingWindow(object):
    def __init__(self, image, width, ratio, overlap=0.85, zoom_factor=lambda x: 1 - x * .1):
        """
        Creates a new scaling sliding window. It can be used as an iterator.
        :param image: The image to use. For best performance, only perform raw_scale on the loaded image.
        :param width: The vertical size of the sliding window in pixels or in percentage,  if <= 1.
        :param ratio: The ratio of the sliding window. If it is 0, the images ratio will be used.
        :param overlap: How much the sliding window will overlap after each step, as fractal.
        :param zoom_factor: A function to determine the next zoom factor. It will get the number of the iteration,
               stating with 0. After each iteration the image will be zoomed to the width and height of the original
               image multiplied with the result of this function. When the function returns a value <= 0 or > 1, no
               further image will be returned.
        """

        # Copy the constructor arguments
        self.image = ScalingSlidingWindow.parse_image(image)
        self.ratio = ratio
        self.overlap = overlap
        self.zoom = zoom_factor

        # Initiate misc attributes
        self.iteration = 0
        self.reached_max_zoom = False
        self.factor_end = False

        # Get the height and width of the image
        self.image_width = self.image.shape[1]
        self.image_height = self.image.shape[0]

        # Create Window to get the sliding box size
        self.window = Window.create(width, self.ratio, self.image_height, self.image_width)

        # Create first sliding window
        self.__zoom()

    def __iter__(self):
        """see iter in SlidingWindow."""
        return self

    def next(self):
        """
        Calculates the next window except. A StopIteration will be risen if no next except is available.
        :return An excerpt of the source image and a region of interest, which points to the region of the excerpt in
                the source image.
        :returns: numpy.ndarray, RegionOfInterest
        """
        try:
            return self.scale_back(self.slidingWindow.next())
        except StopIteration:
            if self.reached_max_zoom:
                raise
            # Create new sliding window
            self.__zoom()
            if self.factor_end:  # If the zoom factor is zero or smaller, the last image has been returned.
                raise
            return self.scale_back(self.slidingWindow.next())

    def __zoom(self):
        """
        Creates a new sliding window with the next zoom factor. It will set self.reached_max_zoom if the new sliding
        window is either at max width or at max height.
        """
        """
        TODO The scaling of the images is currently done by using a transformer, which only works when the image is
        transposed like (y, x, z) which it is after loading. Currently, the image is passed into the scaling sliding
        window transposed like (z, y, x) which it is after using the transformer. Consider either using a different
        scaling method or passing it not transformed into the scaling sliding window.
        """

        # Scale the image
        # 1. Find out how much
        self.factor = self.zoom(self.iteration)
        if self.factor <= 0 > 1:
            self.factor_end = True
            return

        # 3. Create the transformer to scale
        transformer = caffe.io.Transformer(
            {'data': (1, 3, int(round(self.image_height * self.factor)), int(round(self.image_width * self.factor)))}
        )
        # Set transposing to what the sliding window needs
        transformer.set_transpose('data', (2, 0, 1))
        # 5. Finally, scale
        scaled = transformer.preprocess('data', self.image)

        # Create the new sliding window
        self.slidingWindow = SlidingWindow(scaled, self.window.width, self.ratio, self.overlap)

        # Check, if it's zoomed to the maximum
        if self.slidingWindow.window.reached_right or self.slidingWindow.window.reached_bottom:
            self.reached_max_zoom = True

        # Up the iteration
        self.iteration += 1

    def scale_back(self, (image, roi)):
        """
        Takes a SlidingWindow result and scales the RegionOfInterest so, that it represents a region in the source image
        of this ScalingSlidingWindow.
        :return The result properly scaled.
        :returns: numpy.ndarray, RegionOfInterest
        """
        return image, roi.project(1 / self.factor)

    @staticmethod
    def parse_image(image):
        # Check, if the image has been transposed
        if image.shape[0] == 3:
            # 2. Transpose the image to correct format
            warnings.warn("A scaling window received an already transposed image.")
            return image.transpose(1, 2, 0)
        return image


def test():
    # Load the image
    i = caffe.io.load_image('/home/leifb/Downloads/Schilder/50.jpg')

    # load input and configure prepossessing
    transformer = caffe.io.Transformer({'data': (1, 3, 1200, 1200)})
    transformer.set_raw_scale('data', 255.0)
    processed = transformer.preprocess('data', i)

    ssc = ScalingSlidingWindow(processed, .4, 1, 0.6, lambda x: 1 - x * .2)

    transformer2 = caffe.io.Transformer({'data': (1, 3, ssc.window.height, ssc.window.width)})
    transformer2.set_raw_scale('data', 1.0 / 255.0)

    it = 0
    for image, roi in ssc:
        print '%03d: x: %04d y: %04d w: %04d h: %04d)' % (it, roi.x1, roi.y1, roi.width, roi.height)
        transposed = image.transpose(1, 2, 0)
        scaled = transformer2.preprocess('data', transposed)
        imsave('image/%s.bmp' % it, scaled)
        it += 1

# test()
