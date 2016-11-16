from sign_detection.model.Image import Image
from sign_detection.model.RegionOfInterest import RegionOfInterest


class IdentifiedImage(Image):
    def set_region_of_interests(self, value):
        self._region_of_interests = value

    def get_region_of_interests(self):
        return self._region_of_interests

    region_of_interests = property(get_region_of_interests, set_region_of_interests)

    def __init__(self, path_to_image, region_of_interests):
        # type: (str, list(RegionOfInterest)) -> IdentifiedImage
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

