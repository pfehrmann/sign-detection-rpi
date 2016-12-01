import picamera
import picamera.array
import sign_detection.GTSRB.use_net as un

net, transformer = un.load_net("quadruple_nin_deploy.prototxt", "quad_nin.caffemodel")

with picamera.PiCamera() as camera:
    camera.resolution = (480, 360)
    camera.start_preview()

    while True:
        with picamera.array.PiRGBArray(camera) as output:
            camera.capture(output, 'rgb', use_video_port=True)

            net = un.supply_image(output.array, net, transformer)

            result, prop = un.compute(net)

            camera.annotate_text = un.get_name_from_category(result) + ", " + str(prop)