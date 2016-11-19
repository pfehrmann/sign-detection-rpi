import math


class RegionOfInterest:
    x1 = property()
    y1 = property()

    x2 = property()
    y2 = property()

    sign = property()

    def __init__(self, x1, y1, x2, y2, sign):
        """
        :type x1: int
        :type y1: int
        :type x2: int
        :type y2: int
        :type sign: int
        """

        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2

        self.sign = sign

    @x1.setter
    def x1(self, value):
        self._x1 = value

    @x1.getter
    def x1(self):
        return self._x1

    @y1.setter
    def y1(self, value):
        self._y1 = value

    @y1.getter
    def y1(self):
        return self._y1

    @x2.setter
    def x2(self, value):
        self._x2 = value

    @y2.setter
    def y2(self, value):
        self._y2 = value

    @sign.setter
    def sign(self, value):
        self._sign = value

    @sign.getter
    def sign(self):
        return self._sign

    def getOverlap(self, other):
        """
        Calulated the overlap of two regions
        :param other: The region of interest to intersect with
        :return: Returns a percentage of the overlap

        :type other: RegionOfInterest
        """
        overlap = 0
        return 0.01 * max(0, min(self.x2, other.x2) - max(self.x1, other.x1)) \
               * max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
