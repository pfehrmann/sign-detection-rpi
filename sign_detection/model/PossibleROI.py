from sign_detection.model.RegionOfInterest import RegionOfInterest


class PossibleROI(RegionOfInterest):
    probability = property()

    def __init__(self, x1, y1, x2, y2, sign, probability, zoom_factor_x, zoom_factor_y):
        super(PossibleROI, self).__init__(x1, y1, x2, y2, sign)
        self._probability = probability
        self.zoom_factor = (zoom_factor_x, zoom_factor_y)

    @probability.setter
    def probability(self, value):
        self._probability = value

    @probability.getter
    def probability(self):
        return self._probability

    def __str__(self):
        return '(roi:{roi}, prob:{p}, zoom:{z})'.format(
            roi=super(PossibleROI, self).__str__(),
            p=self._probability,
            z=self.zoom_factor)

    @property
    def unscaled(self):
        return PossibleROI(
            self.x1 / self.zoom_factor[0],
            self.y1 / self.zoom_factor[1],
            self.x2 / self.zoom_factor[0],
            self.y2 / self.zoom_factor[1],
            self.sign, self._probability, 1, 1)
