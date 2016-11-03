class Image(object):
    def __init__(self, path):
        """
        :param path: The path to the image
        :type path: String
        """
        self.path = path

    path = property()
