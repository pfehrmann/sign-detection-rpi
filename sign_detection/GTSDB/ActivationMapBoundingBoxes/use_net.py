import caffe
import cv2
import numpy as np
from matplotlib import pyplot as plt

from sign_detection.model.DetectorBase import DetectorBase
from sign_detection.model.PossibleROI import PossibleROI
from sign_detection.model.RegionOfInterest import RegionOfInterest
from sign_detection.model.Sign import get_name_from_category


class Detector(DetectorBase):
    """
    Use this lass to detect signs using a method similar to Faster RCNNs
    """

    def __init__(self, net, minimum=0.99, use_global_max=True, threshold_factor=0.5,
                 draw_results=False, zoom=[1, 2, 3], area_threshold_min=49, area_threshold_max=10000,
                 activation_layer="conv3", out_layer="softmax", display_activation=False, blur_radius=0,
                 size_factor=0.4, max_overlap=0.2, global_pooling_layer="pool1", faster_rcnn=False, average_value=30,
                 modify_average_value=False):
        """

        :param net: The net to use
        :param minimum: The minimum probability of a class. Everything below is discarded
        produced. Should be in range [0, 1).
        :param use_global_max: Use the global maximum or use local maxima of filters?
        :param threshold_factor: The threshold is defined by the maximum * threshold_factor
        :param draw_results: This outputs the found results visibly
        :param zoom: The factors to zoom into the image
        :param area_threshold_min: Minimum size of a region
        :param area_threshold_max: Maximum size of a region
        :param activation_layer: The layer that yields the activation map
        :param out_layer: The layer that yields the class
        :param display_activation: If true all the activations will be displayed
        :param blur_radius: If > 1 the activation map will be blurred by the radius
        :param size_factor: Increasing the boxes by this amount before checking them
        :param max_overlap: The maximum overlap of two bounding boxes in percent befre they are removed
        """
        self.max_overlap = max_overlap
        self.size_factor = size_factor
        self.blur_radius = blur_radius
        self.display_activation = display_activation
        self.out_layer = out_layer
        self.activation_layer = activation_layer
        self.area_threshold_max = area_threshold_max
        self.area_threshold_min = area_threshold_min
        self.zoom = zoom
        self.draw_results = draw_results
        self.threshold_factor = threshold_factor
        self.use_global_max = use_global_max
        self.minimum = minimum
        self.net = net
        self.global_pooling_layer = global_pooling_layer
        self.faster_rcnn = faster_rcnn
        self.average_value = average_value
        self.modify_average_value = modify_average_value
        self.last_activation_maps = None

    def identify_regions_from_image(self, im, unmodified=None):
        """
        Load and process a net and image
        :param unmodified: The unmodified version of the image to draw on
        :param im: The image to work with
        :return: Returns all the found ROIs as a list of PossibleROI elements.
        :returns: (list[PossibleROI], list[PossibleROI])
        """

        if self.modify_average_value:
            im = set_average_value(im, self.average_value)
        else:
            im = im

        # collect all the regions of interest
        unfiltered_rois, all_activation_maps = self.collect_regions(im)
        self.last_activation_maps = all_activation_maps

        # check each roi individually
        self._check_rois(im, unfiltered_rois)

        # filter all the rois with a too low possibility
        rois = [roi for roi, activation_map in unfiltered_rois if roi.probability >= self.minimum]
        unfiltered_rois_ret = [roi for roi, activation_map in unfiltered_rois]

        if self.draw_results and unmodified is not None:
            self.draw_results_to_image(rois, unfiltered_rois_ret, unmodified)
        return rois, unfiltered_rois_ret

    def remove_overlapping_regions(self, set_a, set_b):
        """
        Check if any region of set_a and b are overlapping. Returns a list in which none are overlapping.
        :param set_a: The regions to go trough
        :param set_b: The regions to go trough
        :type set_a: list[PossibleROI]
        :type set_b: list[PossibleROI]
        :return: Returns an array of regions
        returns: list[(PossibleROI, numpy.ndarray)]
        """
        result = set_a[:]
        result.extend(set_b)
        if self.max_overlap >= 1:
            return result

        if len(set_a) > len(set_b):
            small_set = set_a
            large_set = set_b
        else:
            small_set = set_b
            large_set = set_a

        for large in large_set:
            for small in small_set:
                if small == large:
                    continue
                if small.get_overlap(large) > self.max_overlap:
                    if small.area() < large.area:
                        small_set.remove(small)
                        result.remove(small)
                    else:
                        result.remove(large)
                    break
        large_set.extend(small_set)
        return large_set

    def collect_regions(self, im):
        """
        Get regions from all sizes of images
        :param im: The image to check
        :return: Returns a list of Possible Rois and the uncropped activation maps
        :returns: list[(PossibleROI, np.ndarray)]
        """
        overlapping_rois = []
        all_activation_maps = {}
        for step in self.zoom:
            factor = 1.0 / step
            resized = cv2.resize(im, None, fx=factor, fy=factor)
            new_regions, activation_maps = self.identify_regions(resized)

            all_activation_maps[step] = activation_maps

            for new_region in new_regions:
                new_region.x1 *= step
                new_region.x2 *= step
                new_region.y1 *= step
                new_region.y2 *= step
                new_region.zoom_factor = (new_region.zoom_factor[0] * step, new_region.zoom_factor[1] * step)
                overlapping_rois.append((new_region, activation_maps))

        return overlapping_rois, all_activation_maps

    @staticmethod
    def draw_results_to_image(rois, unfiltered_rois, unmodified):
        draw_regions(unfiltered_rois, unmodified, (0, 1, 0))
        draw_regions(rois, unmodified, (0, 0, 1))
        # show the image and delay the execution
        cv2.imshow("ROIs", unmodified)
        cv2.waitKey(1000000)
        # save the image. Needs mapping to [0,255]
        cv2.imwrite("result.png", unmodified * 255.0)

    def filter_rois(self, rois):
        all_regions = [(roi, maps, False) for roi, maps in rois[:]]

        count = 0
        merged = True
        while merged and count < 3 and len(all_regions) > 1:
            merged_rois = []
            merged_rois_indices = np.zeros((len(all_regions), len(all_regions)))
            count += 1
            merged = False
            for i in range(len(all_regions)):
                roi_tuple = all_regions[i]
                roi = roi_tuple[0]  # type: RegionOfInterest

                for j in range(len(all_regions)):
                    if i == j or merged_rois_indices[i, j] != 0:
                        continue
                    other_tuple = all_regions[j]
                    other = other_tuple[0]
                    if roi is not other and roi.get_overlap(other) > self.max_overlap:
                        new_roi = self.merge(roi, other)
                        merged_rois.append((new_roi, roi_tuple[1], False))
                        all_regions[i] = (roi_tuple[0], roi_tuple[1], True)
                        all_regions[j] = (other_tuple[0], other_tuple[1], True)
                        merged_rois_indices[i, j] = 1
                        merged_rois_indices[j, i] = 1
                        merged = True

            if merged:
                all_regions = [x for x in all_regions if not x[2]]
                all_regions.extend(merged_rois)

        return all_regions

    def get_activation(self, image):
        # Transpose to fit caffes needs
        caffe_in = image.transpose((2, 0, 1))

        # store the original shape of the input layer
        original_shape = self.net.blobs['data'].shape

        # reshape the input layer to match the images size
        width = caffe_in .shape[1]
        height = caffe_in .shape[2]
        self.net.blobs['data'].reshape(1, 3, width, height)

        # set the data and forward
        self.net.blobs['data'].data[...] = caffe_in
        out = self.net.forward(blobs=[self.out_layer, self.activation_layer], end=self.activation_layer)

        # reset the shape
        self.net.blobs['data'].reshape(*original_shape)

        return out[self.activation_layer][:]

    def identify_regions(self, image):
        """
        Identify regions in an image.
        :param image: The image array
        :return: A list of rois with probabilities
        :returns: list[PossibleROI]
        """

        # return parameters
        rois = []

        # Transpose to fit caffes needs
        caffe_in = image.transpose((2, 0, 1))

        # store the original shape of the input layer
        original_shape = self.net.blobs['data'].shape

        # reshape the input layer to match the images size
        width = caffe_in.shape[1]
        height = caffe_in.shape[2]
        self.net.blobs['data'].reshape(1, 3, width, height)

        # set the data and forward
        self.net.blobs['data'].data[...] = caffe_in
        out = self.net.forward(blobs=[self.out_layer, self.activation_layer], end=self.activation_layer)

        # get the activation for the proposals from the activation layer
        activation = out[self.activation_layer]
        global_max = activation.max()
        factor_y = image.shape[0] / activation.shape[2]
        factor_x = image.shape[1] / activation.shape[3]

        if self.display_activation:
            self.display_activation_maps(activation)

        for filter_index in range(len(activation[0])):
            # analyze image
            activation_map = activation[0][filter_index]
            regions, contours = self.__get_regions_from_filter(factor_x, factor_y, activation_map, global_max)

            rois = self.remove_overlapping_regions(rois, regions)

        # reset the shape
        self.net.blobs['data'].reshape(original_shape[0], original_shape[1], original_shape[2], original_shape[3])

        # Show some information about the regions
        # print "Number Regions: " + str(len(rois))
        return rois, activation[:]

    @staticmethod
    def display_activation_maps(layer_blob):
        plot = 1
        count_plots = layer_blob.shape[1]
        width = int(count_plots ** 0.5)
        height = width + 1
        for map in layer_blob[0]:
            plt.subplot(width, height, plot), plt.imshow(map, 'gray')
            plot += 1
        plt.show()

    def __get_regions_from_filter(self, factor_x, factor_y, filter, global_max):
        rois = []
        max_value = filter.max()
        if self.use_global_max:
            threshold_value = global_max * self.threshold_factor
        else:
            threshold_value = max_value * self.threshold_factor

        # apply threshold
        ret, thresh = cv2.threshold(np.uint8(filter), threshold_value, 0, cv2.THRESH_TOZERO)

        # blur the image to produce better results
        if self.blur_radius > 0:
            thresh = cv2.blur(thresh, (self.blur_radius, self.blur_radius))

        # extract the contours
        if max_value == 0:
            max_value = 1
        converted = (thresh / max_value * 255.0)
        converted = converted.astype(np.uint8)
        im2, contours, hierarchy = cv2.findContours(converted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # create bounding boxes
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            h = max(abs(w), abs(h))
            w = max(abs(w), abs(h))
            area = w * h * factor_x * factor_y
            if self.area_threshold_min <= area <= self.area_threshold_max:
                # append the found roi to the list of rois
                possible_roi = PossibleROI(x * factor_x, y * factor_y, (x + w) * factor_x, (y + h) * factor_y, -1, 0, factor_x, factor_y)
                possible_roi.additional_info["activation_map"] = filter
                rois.append(possible_roi)
        return rois, contours

    def _check_rois(self, image, rois):
        """

        :param image:
        :param rois:
        :return:
        :type rois: (PossibleROI, int[])[]
        """
        if self.faster_rcnn:
            self._check_rois_faster(rois)
            return

        for roi, activation_maps in rois:
            caffe_in = _prepare_image_for_roi(image, self.net.blobs['data'].shape, roi, self.size_factor)
            self.net.blobs['data'].data[...] = caffe_in

            out = self.net.forward()

            # get the class
            class_index = out[self.out_layer].argmax()
            possibility = out[self.out_layer][0][class_index]

            roi.probability = possibility
            roi.sign = class_index

    def _check_rois_faster(self, rois):
        """
        Check a roi without passing the corresponding image through the net again
        :param rois:
        :return: Nothing
        :type rois: (PossibleROI, int[])[]
        """
        for roi, activation_maps in rois:
            maps = _prepare_activation_maps(maps=activation_maps,
                                            x1=roi.x1 / roi.zoom_factor[0],
                                            x2=roi.x2 / roi.zoom_factor[0],
                                            y1=roi.y1 / roi.zoom_factor[1],
                                            y2=roi.y2 / roi.zoom_factor[1],
                                            size_factor=self.size_factor)

            # Resize global pooling layers input
            self.net.blobs[self.global_pooling_layer].data[...] = maps
            out = self.net.forward(start="ip1_1")

            # get the class
            class_index = out[self.out_layer].argmax()
            possibility = out[self.out_layer][0][class_index]

            roi.probability = possibility
            roi.sign = class_index

    @staticmethod
    def merge(roi, other):
        """
        :type roi: PossibleROI
        :type other: PossibleROI
        :param roi:
        :param other:
        :return:
        """
        return PossibleROI(min(roi.x1, other.x1),
                           min(roi.y1, other.y1),
                           max(roi.x2, other.x2),
                           max(roi.y2, other.y2),
                           roi.sign,
                           roi.probability,
                           roi.zoom_factor[0],
                           roi.zoom_factor[1])


def preprocess_image(image):
    return set_average_value(image, 0.4)


def set_average_value(image, val):
    average = np.average(cv2.mean(image)[:3])
    lut = np.array(range(0, 256)) * (val / average)
    lut = lut.clip(0, 255).astype(np.uint8)
    res = cv2.LUT(src=image.astype(np.uint8), lut=lut)
    return res


def _prepare_image_for_roi(image, original_shape, roi, size_factor):
    crop_img = __crop_image(image, roi, size_factor)
    crop_img = caffe.io.resize_image(crop_img, (original_shape[2], original_shape[3]))
    caffe_in = crop_img.transpose((2, 0, 1))
    return caffe_in


def _prepare_activation_maps(maps, x1, y1, x2, y2, size_factor):
    """

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    :type x1: int
    :type x2: int
    :type y1: int
    :type y2: int
    """
    cropped_maps = _crop_activation_maps(maps, size_factor, x1, x2, y1, y2)
    ret = np.zeros((len(cropped_maps), len(cropped_maps[0]), 1, 1))
    for i_batch, batch in enumerate(cropped_maps):
        for i_filter, activation_map in enumerate(batch):
            ret[i_batch, i_filter, 0, 0] = activation_map.max()
    return ret


def _crop_activation_maps(maps, size_factor, x1, x2, y1, y2):
    region = RegionOfInterest(x1, y1, x2, y2, -1)
    region = __scale_roi(maps[0][0], region, size_factor)
    cropped_maps = maps[:, :, int(region.y1):int(region.y2), int(region.x1):int(region.x2)]
    return cropped_maps


def draw_regions(rois, image, color=(0, 0, 1), print_class=False):
    for roi in rois:

        # In case that the roi is actually a tuple consisting of the roi and the activation maps...
        try:
            roi = roi[0]
        except:
            pass
        cv2.rectangle(image, (int(roi.x1), int(roi.y1)), (int(roi.x2), int(roi.y2)), color=color, thickness=2)
        if print_class:
            retval, base_line = cv2.getTextSize(str(roi.sign), cv2.FONT_HERSHEY_PLAIN, 1, 1)
            dx = retval[0]
            dy = retval[1]
            cv2.rectangle(image, (int(roi.x1), int(roi.y2)), (int(roi.x1 + dx + 2), int(roi.y2 - dy - 2)), color,
                          thickness=cv2.FILLED)
            cv2.putText(image, str(roi.sign), (int(roi.x1 + 1), int(roi.y2 - 1)), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0))


def __crop_image(image, roi, size_factor):
    copy = __scale_roi(image, roi, size_factor)
    crop_img = np.array(image[int(copy.y1):int(copy.y2), int(copy.x1):int(copy.x2)], dtype=np.float16)
    return crop_img


def __scale_roi(image, roi, size_factor):
    copy = RegionOfInterest(roi.x1, roi.y1, roi.x2, roi.y2, roi.sign)
    copy.increase_size(size_factor)
    copy.x1 = max(0, copy.x1)
    copy.y1 = max(0, copy.y1)
    copy.x2 = min(image.shape[1], copy.x2)
    copy.y2 = min(image.shape[0], copy.y2)
    return copy


def draw_contours(image, contours):
    draw = np.array(image / image.max() * 255, dtype=np.uint8)
    draw = cv2.cvtColor(draw, cv2.COLOR_GRAY2RGB)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        h = max(w, h)
        w = max(w, h)
        area = w * h
        if area >= 49:
            cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Regions", draw)
    cv2.waitKey(1000)


def load_net(model, weights):
    # load and return the net
    return caffe.Net(model, weights, caffe.TEST)


def setup_device(gpu=True):
    """
    Sets up caffe computing device
    :param gpu: Use the GPU? If False, CPU is used
    :return: Nothinh
    """
    # use either cpu or gpu
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()


def load_image(image_path, factor=255.0 * 0.3):
    """
    Loads and scales an image
    :param image_path: Path to the image
    :param factor: The factor to scale the image (image = image * factor)
    :return: Both the scaled image and the original
    :returns: [[float]],[[float]]
    """
    # load the image and swap channels
    im = caffe.io.load_image(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # return a scaled version of the image. Necessary to reduce the activation of some neurons...
    return im * factor, im


def identify_regions_from_image_path(model, weights, image_path, gpu=True, minimum=0.99, factor=255.0 * 0.3,
                                     use_global_max=True, threshold_factor=0.5, draw_results=False, zoom=[1, 2, 3],
                                     area_threshold_min=49, area_thrshold_max=10000, activation_layer="conv3",
                                     out_layer="softmax", display_activation=False, blur_radius=0, faster_rcnn=False,
                                     size_factor=0.4):
    """
    Load and process a net and image
    :param blur_radius: If > 1 the activation map will be blurred by the radius
    :param display_activation: If true all the activations will be displayed
    :param out_layer: The layer that yields the class
    :param activation_layer: The layer that yields the activation map
    :param area_thrshold_max: Maximum size of a region
    :param area_threshold_min: Minimum size of a region
    :param zoom: The factors to zoom into the image
    :param model: The path to the prototxt model definition
    :param weights: The path to the caffemodel weights file
    :param image_path: The path to the image
    :param gpu: Use the GPU? Default is true
    :param minimum: The minimum probability of a class. Everything below is discarded
    produced. Should be in range [0, 1).
    :param factor: The facor to multiply the image with. Use this to prevent over stimulation.
    :param use_global_max: Use the global maximum or use local maxima of filters?
    :param threshold_factor: The threshold is defined by the maximum * threshold_factor
    :param draw_results: This outputs the found results visibly
    :return: Returns all the found ROIs as a list of PossibleROI elements.
    :returns: list[PossibleROI]
    """

    # initialize caffe
    setup_device(gpu)

    # initialize net and image
    net = load_net(model, weights)
    im, unmodified = load_image(image_path, factor)
    detector = Detector(net, minimum=minimum,
                        use_global_max=use_global_max,
                        threshold_factor=threshold_factor, draw_results=draw_results, zoom=zoom,
                        area_threshold_min=area_threshold_min,
                        area_threshold_max=area_thrshold_max,
                        activation_layer=activation_layer, out_layer=out_layer,
                        display_activation=display_activation, blur_radius=blur_radius, faster_rcnn=faster_rcnn, size_factor=size_factor)
    return detector.identify_regions_from_image(im=im, unmodified=unmodified)


# parse_arguments()
if __name__ == "__main__":
    _regions, _unfiltered = identify_regions_from_image_path(
        "C:/Users/Philipp/PycharmProjects/sign-detection-playground/sign_detection/GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
        "C:/Users/Philipp/PycharmProjects/sign-detection-playground/sign_detection/GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel",
        "E:/development/GTSDB/FullIJCNN2013/00710.ppm", minimum=0.9999, factor=255.0 * 0.5, use_global_max=False,
        threshold_factor=0.80, draw_results=True, zoom=[0.5, 1, 2], area_threshold_min=1000, area_thrshold_max=50000,
        activation_layer="activation", display_activation=False, gpu=True, blur_radius=0, faster_rcnn=True,
        size_factor=1)

    for _roi in _regions:
        print get_name_from_category(_roi.sign) + " (" + str(_roi.probability) + ") @({},{}), ({},{})".format(_roi.x1,
                                                                                                              _roi.y1,
                                                                                                              _roi.x2,
                                                                                                              _roi.y2)
