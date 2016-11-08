from sign_detection.model.Image import Image


class IdentifiedImage(Image):
    region_of_interests = property()

    def __init__(self, path_to_image, region_of_interests):
        super(IdentifiedImage, self).__init__(path_to_image)
        rois = []
        rois.extend(region_of_interests)
        self.region_of_interests = rois

    @region_of_interests.setter
    def region_of_interests(self, value):
        self._region_of_interests = value

    @region_of_interests.getter
    def region_of_interests(self):
        return self._region_of_interests
