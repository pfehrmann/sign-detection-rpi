import caffe.io
import skimage.io

from sign_detection.model.SlidingWindow import SlidingWindow
from sign_detection.model.Window import Window


class ScalingSlidingWindow(object):
    def __init__(self, image, width, ratio, overlap=0.85, zoom_factor=lambda x: 1 - x * .1):
        self.image = image
        self.ratio = ratio
        self.overlap = overlap
        self.zoom = zoom_factor
        self.iteration = 0
        self.reached_max_zoom = False
        self.image_width = self.image.shape[2]
        self.image_height = self.image.shape[1]

        # Create Window to get the shape size
        self.window = Window.create(width, self.ratio, self.image_height, self.image_width)

        # Create first sliding window
        self.__zoom()

    def __iter__(self):
        return self

    def next(self):
        try:
            return self.slidingWindow.next()
        except StopIteration:
            if self.reached_max_zoom:
                raise
            # Create new sliding window
            self.__zoom()
            return self.slidingWindow.next()

    def __zoom(self):
        factor = self.zoom(self.iteration)
        transposed = self.image.transpose(1, 2, 0)
        transformer = caffe.io.Transformer(
            {'data': (1, 3, int(round(self.image_height * factor)), int(round(self.image_width * factor)))}
        )
        transformer.set_transpose('data', (2, 0, 1))
        scaled = transformer.preprocess('data', transposed)
        self.slidingWindow = SlidingWindow(scaled, self.window.width, self.ratio, self.overlap)

        # Check, if it's zoomed to the maximum
        if self.slidingWindow.window.reached_right or self.slidingWindow.window.reached_bottom:
            self.reached_max_zoom = True

        # Up he iteration
        self.iteration += 1


def test():
    i = caffe.io.load_image('/home/leifb/Downloads/Schilder/Vorfahrt.png')

    # load input and configure prepossessing
    transformer = caffe.io.Transformer({'data': (1, 3, 412, 963)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    processed = transformer.preprocess('data', i)

    ssc = ScalingSlidingWindow(processed, 192, 1, 0.85, lambda x: 1 - x * .2)

    transformer2 = caffe.io.Transformer({'data': (1, 3, ssc.window.height, ssc.window.width)})
    # transformer2.set_channel_swap('data', (2, 1, 0))
    transformer2.set_raw_scale('data', 1.0 / 255.0)

    it = 0
    for image, roi in ssc:
        print 'got image nr %s' % it
        transposed = image.transpose(1, 2, 0)
        scaled = transformer2.preprocess('data', transposed)
        skimage.io.imsave('image/%s.bmp' % it, scaled)
        it += 1

# test()
