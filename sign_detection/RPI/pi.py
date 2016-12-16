import picamera
import picamera.array
import sign_detection.GTSRB.use_net as un
import time

net, transformer = un.load_net("deploy.prototxt", "model.caffemodel")

with picamera.PiCamera() as camera:
    camera.resolution = (1080, 1080)
    camera.start_preview()
    detected = ""
    while True:
        with picamera.array.PiRGBArray(camera) as output:
            t = time.time()
            camera.capture(output, 'rgb', use_video_port=True)

            net = un.supply_image(output.array, net, transformer)

            result, prop = un.compute(net)
            prop = prop[0][0]

            t2 = time.time()

            if (prop > 0.95):
                detected = un.get_name_from_category(result) + ", {0:.2f}".format(prop * 100)
            camera.annotate_text = detected + "\n" + un.get_name_from_category(result) + ", {0:.2f}%".format(
                prop * 100) + "\n@{0:.2f}".format(1. / (t2 - t)) + "fps"