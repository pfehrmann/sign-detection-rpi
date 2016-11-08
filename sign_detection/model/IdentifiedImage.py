from sign_detection.model.Image import Image
from sign_detection.model.RegionOfInterest import RegionOfInterest


class IdentifiedImage(Image):
    region_of_interests = property()

    def __init__(self, path_to_image, region_of_interests):
        """

        :param path_to_image:
        :param region_of_interests:
        :type path_to_image: str
        :type region_of_interests: list[RegionOfInterest]
        """
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
