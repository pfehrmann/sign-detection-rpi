import caffe
import Image


class SlidingWindow(object):
    def __init__(self, image, size, ratio, overlap):
        """
        Creates a new sliding window.
        :param image: The Image to use
        :param size: The vertical size of the sliding window in pixels or in percentage,  if <= 1.
        :param ratio: The ratio of the sliding window. If it is 0, the images ratio will be used.
        :param overlap: How much the sliding window will overlap after each step. 1 means  no overlapping, 0 no moving.
        """

        self.image = image
        self.size = size
        self.ratio = float(ratio)
        self.overlap = float(overlap)

        # load the image
        self.__load_image()

        # init window box
        self.window_pos = [0, 0]
        self.__set_window_size()
        self.step = [int(round(x * self.overlap)) for x in self.window_size]
        self.window_reached_right = False
        self.window_reached_bottom = False

        print 'size: %s\nstep: %s' % (self.window_size, self.step)

    def __load_image(self):
        """
        Loads the image into the memory using caffe and sets image_size.
        """
        self.image_raw = caffe.io.load_image(self.image.path)
        self.image_size = [self.image_raw.shape[1], self.image_raw.shape[0]]

    def __set_window_size(self):
        """
        Calculates window_size by using size, ratio and image_size. The rules are defined in the constructor comment.
        Due to the fact that size can be absolute or relative and the ratio can be taken from the image, multiple
        code paths are possible, which should be tested!
        """

        if self.ratio == 0:  # Use the image's ratio
            if self.size <= 1:  # Size %
                self.window_size = [int(round(x * self.size)) for x in self.image_size]
            else:  # Size absolute
                # TODO Check if the image ratio is calculated correct
                self.window_size = [int(round(self.size)),
                                    int(round(self.size * (float(self.image_size[0]) / self.image_size[1])))]
        else:  # Use given ratio
            if self.size <= 1:  # Size %
                x = self.image_size[0] * self.size
                self.window_size = [int(round(x)),
                                    int(round(x * self.ratio))]
            else:  # Size absolute
                self.window_size = [int(round(self.size)),
                                    int(round(self.size * self.ratio))]


    def __move_window(self):
        # check, if the corner is reached
        if self.window_reached_right:
            # If the corner and bottom is reached, the last element has been found.
            if self.window_reached_bottom:
                raise StopIteration()

            # Find the next y coordinate and check if it exceeds the image
            new_y = self.window_pos[1] + self.step[1]
            if new_y + self.window_size[1] > self.image_size[1]:
                pass
        new_x = self.window_pos[0] + self.step[0]
        if new_x + self.window_size[0] > self.image_size[0]:
            pass


    def __iter__(self):
        return_image = self.image_raw[self.window_pos[1]: self.window_pos[1] + self.window_size[1],
                              self.window_pos[0]: self.window_pos[0] + self.window_size[0],
                              0: 3]
        return return_image



i = Image.Image('/home/leifb/Downloads/Schilder/Vorfahrt.jpg')
s = SlidingWindow(i, 0.5, 0, 0.5)
