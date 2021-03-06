from random import random

from sign_detection.model.Vector import Vector, from_array


class RegionOfInterest(object):
    def __init__(self, x1, y1, x2, y2, sign):
        """
        :type x1: int
        :type y1: int
        :type x2: int
        :type y2: int
        :type sign: int
        """

        self.__p1 = Vector(int(x1), int(y1))
        self.__p2 = Vector(int(x2), int(y2))
        self.__sign = int(sign)

    @property
    def x1(self):
        return self.__p1.x

    @x1.setter
    def x1(self, value):
        self.__p1.x = int(value)

    @property
    def y1(self):
        return self.__p1.y

    @y1.setter
    def y1(self, value):
        self.__p1.y = int(value)

    @property
    def x2(self):
        return self.__p2.x

    @x2.setter
    def x2(self, value):
        self.__p2.x = int(value)

    @property
    def y2(self):
        return self.__p2.y

    @y2.setter
    def y2(self, value):
        self.__p2.y = int(value)

    @property
    def p1(self):
        return self.__p1

    @p1.setter
    def p1(self, value):
        self.__p1 = value

    @property
    def p2(self):
        return self.__p2

    @p2.setter
    def p2(self, value):
        self.__p2 = value

    @property
    def sign(self):
        return self.__sign

    @sign.setter
    def sign(self, value):
        self.__sign = value

    @property
    def size(self):
        return (self.p2 - self.p1).as_array

    @size.setter
    def size(self, size):
        self.p2 = self.p1 + from_array(size)

    @property
    def position(self):
        return self.p1.as_array

    @position.setter
    def position(self, value):
        v = from_array(value)
        self.p2 += + v - self.__p1
        self.p1 = v

    @property
    def width(self):
        return self.p2.x - self.p1.x

    @property
    def height(self):
        return self.p2.y - self.p1.y

    def move(self, v):
        vec = from_array(v)
        self.p1 += vec
        self.p2 += vec
        return self

    def get_overlap(self, other):
        """
        Calculates the overlap of two regions
        :param other: The region of interest to intersect with
        :return: Returns a percentage of the overlap

        :type other: RegionOfInterest
        """
        return self.intersection_over_union(other)

    def is_overlapping(self, region):
        """
        Check if this bounding box overlaps with the given region
        :param region: The region to check overlapping with
        :return: Returns True, if both are overlapping. Returns False otherwise.
        :returns: bool
        """
        if self.x2 < region.x1:
            return False  # this box is left the other
        if self.x1 > region.x2:
            return False  # this box is right the other
        if self.y2 < region.y1:
            return False  # this box is above the other
        if self.y1 > region.y2:
            return False  # this box is below the other
        return True

    def intersection_over_union(self, box_b):
        # if the boxes are not overlapping, then return 0
        if not self.is_overlapping(box_b):
            return 0

        # determine the (x, y)-coordinates of the intersection rectangle
        s_x1 = self.x1
        s_x2 = self.x2
        s_y1 = self.y1
        s_y2 = self.y2

        o_x1 = box_b.x1
        o_x2 = box_b.x2
        o_y1 = box_b.y1
        o_y2 = box_b.y2

        xA = max(s_x1, o_x1)
        yA = max(s_y1, o_y1)
        xB = min(s_x2, o_x2)
        yB = min(s_y2, o_y2)

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (s_x2 - s_x1 + 1) * (s_y2 - s_y1 + 1)
        boxBArea = (o_x2 - o_x1 + 1) * (o_y2 - o_y1 + 1)

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
        return self.width * self.height

    def similar(self, other, min_overlap):
        return self.get_overlap(other) > min_overlap

    def project(self, factor):
        """
        Projects the region of interest onto a area by the given factor. Each coordinate will be scaled by it.
        :param factor: The factor in which the project-to areas differs from the current.
        :return: A new projected RegionOfInterest.
        """
        return RegionOfInterest(self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor, self.sign)

    def clone(self):
        return RegionOfInterest(self.x1, self.y1, self.x2, self.y2, self.sign)

    def add_padding(self, size):
        v = Vector(size, size)
        self.p1 -= v
        self.p2 += v
        return self

    def disturb(self, move_by=0.4, stretch_by=0.3):
        move = self.size
        move[0] = int(move[0] * (random() * 2 - 1) * move_by)
        move[1] = int(move[1] * (random() * 2 - 1) * move_by)
        self.move(move)
        stretch = self.size
        stretch[0] = int(stretch[0] * (random() * 2 - 1) * stretch_by / 2)
        stretch[1] = int(stretch[1] * (random() * 2 - 1) * stretch_by / 2)
        stretch_arr = from_array(stretch)
        self.__p1 -= stretch_arr
        self.__p2 += stretch_arr
        return self

    def clip(self, max_x, max_y, min_x=0, min_y=0):
        self.x1 = max(self.x1, min_x)
        self.y1 = max(self.y1, min_y)
        self.x2 = min(self.x2, max_x)
        self.y2 = min(self.y2, max_y)
        return self

    @property
    def center(self):
        return from_array([self.__p1.x + self.size[0] * 0.5, self.__p1.y + self.size[1] * 0.5])

    def get_distance(self, roi2):
        return [b - a for a, b in zip(self.center.as_array, roi2.center.as_array)]

    def __str__(self):
        return '(p1:{p1}, p2:{p2}, sign:{sign:02d})'.format(p1=str(self.p1), p2=str(self.p2), sign=self.sign)
