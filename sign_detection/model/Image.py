class Image(object):
    path = property()

    def __init__(self, path):
        """
        :param path: The path to the image
        :type path: String
        """
        self.path = path

    @path.setter
    def path(self, value):
        self._path = value

    @path.getter
    def path(self):
        return self._path
