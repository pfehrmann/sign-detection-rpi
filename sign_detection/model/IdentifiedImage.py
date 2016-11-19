from sign_detection.model.Image import Image
from sign_detection.model.RegionOfInterest import RegionOfInterest


class IdentifiedImage(Image):
    def set_region_of_interests(self, rois):
        """
        Set the regions of interest

        :param rois: a list containing regions of interest
        :return: None

        :type rois: list[RegionOfInterest]
        """
        self._region_of_interests = rois

    def get_region_of_interests(self):
        """

        :return: All teh regions of interest
        :returns: list[RegionOfInterest]
        """
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

    def get_overlapping_regions(self, region_of_interest, overlap=0.85):
        """
        Get all Regions Of Interest, that overlap with the given ROI
        :param region_of_interest: The ROI to check
        :param overlap: The minimal percentage of overlap. 0 will return every region, 1 will return only regions that overlap completely.
        :return: A list of all overlapping regoins
        :returns: list[RegionOfInterest]
        :type region_of_interest: RegionOfInterest
        :type overlap: float
        """

        regions = []
        for roi in self.region_of_interests:
            if roi.getOverlap(region_of_interest) > overlap:
                regions.append(roi)
        return regions
