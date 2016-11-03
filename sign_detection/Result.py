class Result(object):
    image = property()
    list_of_rois = property()
    roi_chooser = property()

    def __init__(self, image, list_of_rois, roi_chooser):
        self._image = image
        self._list_of_rois = list_of_rois
        self._roi_chooser = roi_chooser

    @image.setter
    def image(self, value):
        self._image = value

    @list_of_rois.setter
    def list_of_rois(self, value):
        self._list_of_rois = value

    @roi_chooser.setter
    def roi_chooser(self, value):
        self._roi_chooser = value
