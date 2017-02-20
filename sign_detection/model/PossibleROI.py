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
