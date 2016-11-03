from sign_detection.RegionOfInterest import ROI


class PossibleROI(ROI):
    probability = property()

    def __init__(self, x1, y1, x2, y2, sign, probability):
        super(PossibleROI, self).__init__(x1, y1, x2, y2, sign)
        self._probability = probability

    @probability.setter
    def probability(self, value):
        self._probability = value
