import picamera
import picamera.array
import datetime
import os
import errno
import time

def create_directory(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write("FOOBAR")


folder = "/media/pi/Usb/" + str(datetime.datetime.now()) + "/"
create_directory(folder + "bla")

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 25
    while True:
        start = time.time()
        camera.start_recording(folder + str(datetime.datetime.now()) + '.h264')
        time.sleep(10)
        camera.stop_recording()
