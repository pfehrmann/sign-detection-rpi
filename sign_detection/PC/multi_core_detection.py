import time

import sign_detection.GTSDB.multi_processor_detection as mpd
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from sign_detection.GTSDB.image_sources import CV2CameraImageSource, CV2VideoImageSource
from sign_detection.GTSDB.roi_result_handlers import ViewHandler, ConsoleHandler, JunkRemover

num_workers = 4


def create_detector():
    """
    Create a detector
    :return: A Detector
    :returns: sign_detection.GTSDB.Detector_Base
    """
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net_much_junk/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/mini_net_much_junk/weights.caffemodel")
    un.setup_device(gpu=True)
    detector = un.Detector(net,
                           minimum=0.80,
                           use_global_max=False,
                           threshold_factor=0.75,
                           draw_results=False,
                           zoom=[1, 2],
                           area_threshold_min=1000,
                           area_threshold_max=500000,
                           activation_layer="activation",
                           out_layer="softmax",
                           global_pooling_layer="pool1",
                           display_activation=False,
                           blur_radius=1,
                           size_factor=0.5,
                           max_overlap=0.5,
                           faster_rcnn=True,
                           modify_average_value=True,
                           average_value=20)
    return detector


def test():
    """
    Sets up the master and starts detecting.
    :return:
    """
    global num_workers
    # image_source = CV2CameraImageSource()
    image_source = CV2VideoImageSource(rotate_angle=180, saturation_factor=1,
                                       image_source="C:/development/Videos/2017-03-05 23.33.33.437823.h264")
    master = mpd.Master(create_detector, image_source, num_workers=num_workers, method=mpd.BURST)
    view_handler = ViewHandler(num_workers, "Filtered")
    master.register_roi_result_handler(JunkRemover(view_handler, [43]))
    master.register_roi_result_handler(ViewHandler(num_workers, "Unfiltered"))
    master.register_roi_result_handler(JunkRemover(ConsoleHandler(num_workers)))
    master.start()
    try:
        while 1:
            time.sleep(1)
    except KeyboardInterrupt:
        master.stop()

if __name__ == "__main__":
    test()
