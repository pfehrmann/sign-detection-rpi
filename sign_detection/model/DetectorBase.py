class DetectorBase(object):
    """
    Class that defines the basic functionality of detectors.
    """
    def identify_regions_from_image(self, im, unmodified=None):
        """
        Identify all regions of intrest from an image using the Specific method of the class.
        :param im: The preprocessed image to identify regions
        :param unmodified: An unmodified version of the image that can be viewed. This is used to draw on the image.
        :return: Returns both the regions of interest that are considered to be signs and the regions that might be signs.
        :returns: (list[sign_detection.model.PossibleROI.PossibleROI], list[sign_detection.model.PossibleROI.PossibleROI])
        """
        raise NotImplementedError("Should have implemented this")
