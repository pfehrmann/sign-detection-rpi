# pass image through net
# get activation map
# get bbs for each map
# draw image

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from sign_detection.evaluation.GTSDB_evaluation import load as load_image
from sign_detection.model.Image import Image
import cv2
import numpy as np
from skimage.transform import resize
from matplotlib import pyplot as plt

un.setup_device(gpu=True)

# Setup the net and transformer
path = "mini_net"
net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/" + path + "/deploy.prototxt",
                  "../GTSDB/ActivationMapBoundingBoxes/" + path + "/weights.caffemodel")

# setup the detector
detector = un.Detector(net,
                       minimum=0.99999,
                       use_global_max=False,
                       threshold_factor=0.75,
                       draw_results=False,
                       zoom=[1],
                       area_threshold_min=400,
                       area_threshold_max=50000,
                       activation_layer="activation",
                       out_layer="softmax",
                       display_activation=False,
                       blur_radius=0,
                       size_factor=0.1,
                       max_overlap=1,
                       faster_rcnn=True,
                       modify_average_value=True,
                       average_value=70)

# get the image for the net
image = Image(path="C:/development/FullIJCNN2013/FullIJCNN2013/00028.ppm")
image_raw = load_image(image)*255.0

regions, possible_regions = detector.identify_regions_from_image(image_raw)

all_activation_maps = []
for factor, activation_maps in detector.last_activation_maps.iteritems():
    for dimension in activation_maps:
        for activation_map in dimension:
            all_activation_maps.append((activation_map, [], [], factor))

def draw_region_to_filter(region, filter, color, factor):
    x1 = region.x1 / region.zoom_factor[0] * factor
    x2 = region.x2 / region.zoom_factor[0] * factor
    y1 = region.y1 / region.zoom_factor[1] * factor
    y2 = region.y2 / region.zoom_factor[1] * factor
    cv2.rectangle(filter, (x1, y1), (x2, y2), thickness=1, color=color)

def apply_threshold(activation_map, global_max):
    max_value = activation_map.max()
    if detector.use_global_max:
        threshold_value = global_max * detector.threshold_factor
    else:
        threshold_value = max_value * detector.threshold_factor

    # apply threshold
    ret, thresh = cv2.threshold(np.uint8(activation_map), threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

for region in regions:
    activation_map = region.additional_info["activation_map"]

    for map_, roi_list, possible_roi_list, factor in all_activation_maps:
        if not isinstance(map_ == activation_map, bool) and (map_ == activation_map).all() or (isinstance(map_ == activation_map, bool) and (map_ == activation_map)):
            roi_list.append(region)

for region in possible_regions:
    activation_map = region.additional_info["activation_map"]

    for map_, roi_list, possible_roi_list, factor in all_activation_maps:
        if not isinstance(map_ == activation_map, bool) and (map_ == activation_map).all() or (isinstance(map_ == activation_map, bool) and (map_ == activation_map)):
            possible_roi_list.append(region)

# Get the bounds for scaling the maps
max_value = max([activation_map.max() for activation_map, roi_list, possible_roi_list, factor in all_activation_maps])
min_value = min([activation_map.min() for activation_map, roi_list, possible_roi_list, factor in all_activation_maps])

scale_factor = 2
all_activation_maps = [(activation_map, roi_list, possible_roi_list, factor) for activation_map, roi_list, possible_roi_list, factor in all_activation_maps if len(possible_roi_list) > 0]
all_activation_maps = [(apply_threshold(activation_map, max_value), roi_list, possible_roi_list, factor) for activation_map, roi_list, possible_roi_list, factor in all_activation_maps]
max_value = max([activation_map.max() for activation_map, roi_list, possible_roi_list, factor in all_activation_maps])
all_activation_maps = [((activation_map - min_value) / (max_value-min_value), roi_list, possible_roi_list, factor) for activation_map, roi_list, possible_roi_list, factor in all_activation_maps]
all_activation_maps = [(resize(np.float32(activation_map), (activation_map.shape[0] * factor * scale_factor, activation_map.shape[1] * factor * scale_factor)), roi_list, possible_roi_list, factor) for activation_map, roi_list, possible_roi_list, factor in all_activation_maps]
all_activation_maps = [(cv2.cvtColor(np.float32(activation_map), cv2.COLOR_GRAY2RGB), roi_list, possible_roi_list, factor) for activation_map, roi_list, possible_roi_list, factor in all_activation_maps]

for activation_map, roi_list, possible_roi_list, factor in all_activation_maps:
    for possible_roi in possible_regions:
        draw_region_to_filter(possible_roi, activation_map, (0,0,1), 1 * scale_factor) #3-factor)

    for possible_roi in possible_roi_list:
        draw_region_to_filter(possible_roi, activation_map, (0, 1, 0), 1 * scale_factor) #3-factor)

plot = 1
count_plots = len(all_activation_maps) + 1
width = int(count_plots ** 0.5) + 1
height = count_plots / width + 1
for map, roi_list, possible_roi_list, factor in all_activation_maps:
    # plt.subplot(width, height, plot), plt.imshow(map)  # Create a subplot of all maps
    # plot += 1
    plt.imshow(map)  # Create s single plot for each map
    plt.figure()
image_raw /= 255.0
un.draw_regions(possible_regions, image_raw, (0, 1, 0))
un.draw_regions(regions, image_raw, (0, 0, 1), print_class=True)
# plt.subplot(width, height, plot), plt.imshow(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB))
cv2.imshow("", image_raw)
plt.show()
cv2.waitKey(1000000)
