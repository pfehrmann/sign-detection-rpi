import sys
caffe_path = "/home/pi/development/caffe/build/install/python"
if not caffe_path in sys.path:
    sys.path.append(caffe_path)

from sign_detection.GTSDB.image_sources import PiCameraImageSource
from sign_detection.GTSDB.roi_result_handlers import EV3Handler, ConsoleHandler

import time

from sign_detection.GTSDB.multi_processor_detection import Master
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un

num_workers = 5


def create_detector():
    """
    Create a detector
    :return: A Detector
    :returns: sign_detection.GTSDB.Detector_Base
    """
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel")
    un.setup_device(gpu=False)
    detector = un.Detector(net,
                           minimum=0.9999,
                           use_global_max=True,
                           threshold_factor=0.75,
                           draw_results=False,
                           zoom=[1],
                           area_threshold_min=1000,
                           area_threshold_max=500000,
                           activation_layer="activation",
                           out_layer="softmax",
                           display_activation=False,
                           blur_radius=1,
                           size_factor=0.4,
                           max_overlap=0.5,
                           faster_rcnn=True,
                           modify_average_value=False,
                           average_value=60)
    return detector


def test():
    """
    Sets up the master and starts detecting.
    :return:
    """
    global num_workers
    image_source = PiCameraImageSource()
    master = Master(create_detector, image_source, num_workers)
    master.register_roi_result_handler(EV3Handler())
    master.register_roi_result_handler(ConsoleHandler(num_workers))
    master.start()
    try:
        while 1:
            time.sleep(1)
    except KeyboardInterrupt:
        master.stop()


if __name__ == "__main__":
    test()
