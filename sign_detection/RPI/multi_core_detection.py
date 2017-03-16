import sys
caffe_path = "/home/pi/development/caffe/build/install/python"
if not caffe_path in sys.path:
    sys.path.append(caffe_path)

import sign_detection.EV3.EV3 as ev3
from sign_detection.RPI.PiCameraImageSource import PiCameraImageSource
from sign_detection.GTSDB.roi_result_handlers import EV3Handler, ConsoleHandler, ViewHandler

import time

import sign_detection.GTSDB.multi_processor_detection as mpd
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
                           use_global_max=False,
                           threshold_factor=0.75,
                           draw_results=False,
                           zoom=[1, 2],
                           area_threshold_min=100,
                           area_threshold_max=500000,
                           activation_layer="activation",
                           out_layer="softmax",
                           display_activation=False,
                           blur_radius=1,
                           size_factor=0.1,
                           max_overlap=0.35,
                           faster_rcnn=True,
                           modify_average_value=False,
                           average_value=100)
    return detector


def test():
    """
    Sets up the master and starts detecting.
    :return:
    """
    global num_workers
    image_source = PiCameraImageSource(average_value=100)
    master = mpd.Master(create_detector, image_source, num_workers=num_workers, method=mpd.BURST)
    my_ev3 = ev3.EV3(protocol=ev3.USB, host='00:16:53:47:92:46')
    my_ev3.sync_mode = ev3.ASYNC
    master.register_roi_result_handler(EV3Handler(my_ev3))
    master.register_roi_result_handler(ConsoleHandler(num_workers, num_timestamps=num_workers * 10))
    master.register_roi_result_handler(ViewHandler(num_workers))
    master.start()
    try:
        while 1:
            time.sleep(1)
    except KeyboardInterrupt:
        master.stop()


if __name__ == "__main__":
    test()
