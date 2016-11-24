
class RegionOfInterest(object):

    def __init__(self, x1, y1, x2, y2, sign):
        """
        :type x1: int
        :type y1: int
        :type x2: int
        :type y2: int
        :type sign: int
        """

        self.__x1 = x1
        self.__y1 = y1

        self.__x2 = x2
        self.__y2 = y2

        self.__sign = sign

    @property
    def x1(self):
        return self.__x1

    @x1.setter
    def x1(self, value):
        self.__x1 = value

    @property
    def y1(self):
        return self.__y1

    @y1.setter
    def y1(self, value):
        self.__y1 = value

    @property
    def x2(self):
        return self.__x2

    @x2.setter
    def x2(self, value):
        self.__x2 = value

    @property
    def y2(self):
        return self.__y2

    @y2.setter
    def y2(self, value):
        self.__y2 = value

    @property
    def sign(self):
        return self.__sign

    @sign.setter
    def sign(self, value):
        self.__sign = value

    @property
    def size(self):
        return [self.x2 - self.x1, self.y2 - self.y1]

    @size.setter
    def size(self, size):
        self.x2 = self.x1 + size[0]
        self.y2 = self.y1 + size[1]

    @property
    def position(self):
        return [self.x1, self.y2]

    @position.setter
    def position(self, value):
        self.x2 += value[0] - self.x1
        self.y2 += value[1] - self.y1
        self.x1 = value[0]
        self.y1 = value[1]

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

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
