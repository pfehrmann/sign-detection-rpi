import cv2

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from time import time


def identify_regions(save=False, gpu=True):
    # initialize caffe
    un.setup_device(gpu=gpu)

    if save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 2.5, (640, 480))

    # Setup the net and transformer
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
                      "../GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel")
    average_value = 30
    # setup the detector
    detector = un.Detector(net, minimum=0.9999, use_global_max=False, threshold_factor=0.75, draw_results=False,
                           zoom=[1], area_threshold_min=1200, area_threshold_max=50000, activation_layer="activation",
                           out_layer="softmax", display_activation=False, blur_radius=1, size_factor=0.5,
                           faster_rcnn=True, modify_average_value=True, average_value=average_value)

    # capture from camera at location 0
    cap = cv2.VideoCapture(0)

    # Print some of the properties of the camera. For adjustment of speed.
    print "cv2.CAP_PROP_EXPOSURE:   " + str(cap.get(cv2.CAP_PROP_EXPOSURE))
    print "cv2.CAP_PROP_APERTURE:   " + str(cap.get(cv2.CAP_PROP_APERTURE))
    print "cv2.CAP_PROP_BRIGHTNESS: " + str(cap.get(cv2.CAP_PROP_BRIGHTNESS))
    print "cv2.CAP_PROP_CONTRAST:   " + str(cap.get(cv2.CAP_PROP_CONTRAST))
    print "cv2.CAP_PROP_SATURATION: " + str(cap.get(cv2.CAP_PROP_SATURATION))

    # Change the camera setting using the set() function
    cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)  # set exposure so we don't have to scale the image
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, True)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 94.0)
    cap.set(cv2.CAP_PROP_SATURATION, 56.0)
    cap.set(cv2.CAP_PROP_CONTRAST, 24.0)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, True)  # set convert to rgb
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    while True:
        start = time()

        # capture the image
        ret, img = cap.read()
        unmodified_image = img[:]
        # pass the image through the net
        rois, unfiltered = detector.identify_regions_from_image(img, img)

        end = time()

        # Show the regions
        img = un.set_average_value(unmodified_image, average_value)
        un.draw_regions(unfiltered, img, (0, 255, 0))
        un.draw_regions(rois, img, (0, 0, 255), print_class=True)
        cv2.putText(img, "{} fps".format(1.0 / (end - start)), (5, img.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 1)
        cv2.imshow("Detection", img)

        if save:
            out.write(img)

        # Exit with the escape key
        key = cv2.waitKey(10)
        if key == 27:
            break

    # clean up
    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()

    if save:
        out.release()


if __name__ == '__main__':
    identify_regions(save=False, gpu=True)
