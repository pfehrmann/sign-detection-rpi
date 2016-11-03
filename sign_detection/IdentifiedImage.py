from sign_detection.Image import Image


class IdentifiedImage(Image):
    def __init__(self, path_to_image, region_of_interests):
        super(IdentifiedImage, self).__init__(path_to_image)
        self.region_of_interests = region_of_interests

    region_of_interests = property()
