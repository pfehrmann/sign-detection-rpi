import cv2

import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un
from time import time


def identify_regions():
    # initialize caffe
    un.setup_device(gpu=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 2.5, (640, 480))

    # Setup the net and transformer
    net = un.load_net("../GTSDB/ActivationMapBoundingBoxes/mini_net/deploy.prototxt", "../GTSDB/ActivationMapBoundingBoxes/mini_net/weights.caffemodel")

    # capture from camera at location 0
    cap = cv2.VideoCapture(0)

    # Change the camera setting using the set() function
    # see the opencv documentation for a definition of the constants
    cap.set(15, -4.0)  # set exposure so we don't have to scale the image
    cap.set(5, 60)  # set framerate
    cap.set(16, True)  # set convert to rgb

    while True:
        start = time()

        # capture the image
        ret, img = cap.read()

        # pass the image through the net
        rois, unfiltered = un.identify_regions_from_image(img, img, net, minimum=0.9999, use_global_max=True, threshold_factor=0.60,
                                              draw_results=False, zoom=[1, 2], area_threshold_min=1000,
                                              area_thrshold_max=30000, activation_layer="activation", out_layer="softmax",
                                              display_activation=False, blur_radius=1, size_factor=0.4)

        end = time()

        # Show the regions
        un.draw_regions(unfiltered, img, (0, 255, 0))
        un.draw_regions(rois, img, (0, 0, 255))
        cv2.putText(img, "{} fps".format(1.0/(end-start)), (5, img.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.imshow("Detection", img)

        out.write(img)

        # Exit with the escape key
        key = cv2.waitKey(10)
        if key == 27:
            break

    # clean up
    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()
    out.release()


if __name__ == '__main__':
    identify_regions()
