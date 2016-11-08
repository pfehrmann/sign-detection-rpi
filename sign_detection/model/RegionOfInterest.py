class RegionOfInterest:
    x1 = property()
    y1 = property()

    x2 = property()
    y2 = property()

    def __init__(self, x1, y1, x2, y2, sign):
        """
        :type x1: int
        :type y1: int
        :type x2: int
        :type y2: int
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
