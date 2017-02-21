import picamera
import picamera.array
import cv2

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from time import time
import sys


def trace_calls(frame, event, arg):
    if event not in ['call', 'c_call']:
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'write':
        # Ignore write() calls from print statements
        return
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    caller = frame.f_back
    if caller is None:
        return
    caller_line_no = caller.f_lineno
    caller_filename = caller.f_code.co_filename
    print 'Call to %s on line %s of %s from line %s of %s' % \
          (func_name, func_line_no, func_filename,
           caller_line_no, caller_filename)
    return


# sys.settrace(trace_calls)


def identify_regions(save=False, display=True, resultion=(640, 480)):
    # initialize caffe
    un.setup_device(gpu=False)

    if save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 2.5, resultion)

    # Setup the net and transformer
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel")

    # setup the detector
    detector = un.Detector(net, minimum=0.9999, use_global_max=False, threshold_factor=0.75, draw_results=False,
                           zoom=[1], area_threshold_min=2000, area_threshold_max=30000, activation_layer="activation",
                           out_layer="softmax", display_activation=False, blur_radius=1, size_factor=0.4)

    with picamera.PiCamera() as camera:
        camera.resolution = resultion

        while True:
            start = time()

            # capture the image
            output = picamera.array.PiRGBArray(camera)
            camera.capture(output, 'rgb', use_video_port=False)
            img = output.array

            # Flip R and B channel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pass the image through the net
            rois, unfiltered = detector.identify_regions_from_image(img, img)

            end = time()

            # Show the regions
            un.draw_regions(unfiltered, img, (0, 255, 0))
            un.draw_regions(rois, img, (0, 0, 255), print_class=True)
            cv2.putText(img, "{} fps".format(1.0 / (end - start)), (5, img.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 0), 1)

            if display:
                cv2.imshow("Detection", img)

            if save:
                out.write(img)

            # Exit with the escape key
            key = cv2.waitKey(10)
            if key == 27:
                break

    # clean up
    cv2.destroyAllWindows()

    if save:
        out.release()


if __name__ == '__main__':
    identify_regions(save=False, display=True)
