class RegionOfInterest(object):
    def __init__(self, x1, y1, x2, y2, sign):
        """
        :type x1: int
        :type y1: int
        :type x2: int
        :type y2: int
        :type sign: int
        """

        self.__x1 = int(x1)
        self.__y1 = int(y1)

        self.__x2 = int(x2)
        self.__y2 = int(y2)

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

    def get_overlap(self, other):
        """
        Calculates the overlap of two regions
        :param other: The region of interest to intersect with
        :return: Returns a percentage of the overlap

        :type other: RegionOfInterest
        """
        area = float(max(self.area(), other.area()))
        overlap = max(0, min(self.x2, other.x2) - max(self.x1, other.x1)) \
                  * max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return overlap / area

    def intersection_over_union(self, box_b):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.x1, box_b.x1)
        yA = max(self.y1, box_b.y1)
        xB = min(self.x2, box_b.x2)
        yB = min(self.y2, box_b.y2)

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (self.x2 - self.x1 + 1) * (self.y2 - self.y1 + 1)
        boxBArea = (box_b.x2 - box_b.x1 + 1) * (box_b.y2 - box_b.y1 + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def increase_size(self, factor):
        dx = self.width * factor / 2
        dy = self.height * factor / 2
        self.x1 -= dx
        self.x2 += dx
        self.y1 -= dy
        self.y2 += dy

    def area(self):
        return (self.x1 - self.x2) * (self.y1 - self.y2)

    def similar(self, other, min_overlap):
        return self.get_overlap(other) < min_overlap

    def project(self, factor):
        """
        Projects the region of interest onto a area by the given factor. Each coordinate will be scaled by it.
        :param factor: The factor in which the project-to areas differs from the current.
        :return: A new projected RegionOfInterest.
        """
        return RegionOfInterest(self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor, self.sign)
