from sign_detection.model.RegionOfInterest import RegionOfInterest


class Window(RegionOfInterest):

    def __init__(self, x1, y1, x2, y2, reached_right=False, reached_bottom=False):
        super(Window, self).__init__(x1, y1, x2, y2, -1)

        self.__reached_right = reached_right
        self.__reached_bottom = reached_bottom

    @staticmethod
    def create(size, ratio, image_width, image_height):
        """
        Calculates window_size by using size, ratio and image_size. The rules are defined in the constructor comment.
        Due to the fact that size can be absolute or relative and the ratio can be taken from the image, multiple
        code paths are possible, which should be tested!
        """
        # Create a new Window
        window = Window(0, 0, 0, 0)

        # Get size
        if ratio == 0:  # Use the image's ratio
            if size <= 1:  # Size %
                window.x2 = int(round(image_width * size))
                window.y2 = int(round(image_height * size))
            else:  # Size absolute
                window.x2 = int(round(size))
                window.y2 = int(round(size * (float(image_height) / image_width)))
        else:  # Use given ratio
            if size <= 1:  # Size %
                window.x2 = int(round(image_width * size))
                window.y2 = int(round(image_width * size * ratio))
            else:  # Size absolute
                window.x2 = int(round(size))
                window.y2 = int(round(size * ratio))

        # Check, if the window size is larger than the image itself
        window.reached_right = window.width >= image_width
        window.reached_bottom = window.height >= image_height

        # If so, reduce size
        if window.reached_right:
            window.x2 = image_width
        if window.reached_bottom:
            window.y2 = image_height

        return window

    @property
    def reached_bottom(self):
        return self.__reached_bottom

    @reached_bottom.setter
    def reached_bottom(self, value):
        self.__reached_bottom = value

    @property
    def reached_right(self):
        return self.__reached_right

    @reached_right.setter
    def reached_right(self, value):
        self.__reached_right = value

