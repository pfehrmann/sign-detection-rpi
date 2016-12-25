import cv2

import sign_detection.GTSRB.use_net as un
from time import time


def identify_image():
    # Use CPU
    un.setup_device(False)

    # Setup the net and transformer
    net, transformer = un.load_net("deploy.prototxt", "weights.caffemodel")

    # capture from camera at location 0
    cap = cv2.VideoCapture(0)

    # Change the camera setting using the set() function
    # see the opencv documentation for a definition of the constants
    cap.set(15, -5.0)  # set exposure so we don't have to scale the image
    cap.set(5, 60)  # set framerate
    cap.set(16, True)  # set convert to rgb

    detected = ""
    while True:
        start = time()

        # capture the image
        ret, img = cap.read()

        # pass the image through the net
        net = un.supply_image(img, net, transformer)
        result, prop = un.compute(net, out_layer="softmax")
        end = time()

        # if a threshold is reached output the class
        if prop > 0.9999:
            detected = un.get_name_from_category(result) + ", {0:.2f}".format(prop * 100)
        text = detected + "\n" + un.get_name_from_category(result) + ", {0:.2f}%".format(
            prop * 100) + "\n@{0:.2f}".format(1. / (end - start)) + "fps"

        # write the text to the image
        y0, dy = 30, 30
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(img=img, text=line, org=(10, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                        color=(255, 255, 255), thickness=2)

        # display the annotated image
        cv2.imshow("input", img)

        # Exit with the escape key
        key = cv2.waitKey(10)
        if key == 27:
            break

    # clean up
    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()


if __name__ == '__main__':
    identify_image()
