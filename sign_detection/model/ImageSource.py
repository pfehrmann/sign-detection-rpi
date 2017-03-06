class ImageSource(object):
    def get_next_image(self):
        """
        Returns the next image.
        :return: The next image.
        :returns: numpy.ndarray
        """
        raise NotImplementedError()
