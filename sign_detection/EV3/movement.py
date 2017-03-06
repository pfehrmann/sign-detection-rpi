import sign_detection.EV3.EV3 as ev3
import time

myEV3 = ev3.EV3(protocol=ev3.USB, host='00:16:53:47:92:46')


def move(speed, turn):
    global myEV3
    if turn > 0:
        speed_right = speed
        speed_left = round(speed * (1 - turn / 100.0))
    else:
        speed_right = round(speed * (1 + turn / 100.0))
        speed_left = speed
    ops = b''.join([
        ev3.opOutput_Speed,
        ev3.LCX(0),  # LAYER
        ev3.LCX(ev3.PORT_B),  # NOS
        ev3.LCX(speed_right),  # SPEED
        ev3.opOutput_Speed,
        ev3.LCX(0),  # LAYER
        ev3.LCX(ev3.PORT_C),  # NOS
        ev3.LCX(speed_left),  # SPEED
        ev3.opOutput_Start,
        ev3.LCX(0),  # LAYER
        ev3.LCX(ev3.PORT_B + ev3.PORT_C)  # NOS
    ])
    myEV3.send_direct_cmd(ops)


def stop():
    global myEV3
    ops = b''.join([
        ev3.opOutput_Stop,
        ev3.LCX(0),  # LAYER
        ev3.LCX(ev3.PORT_B + ev3.PORT_C),  # NOS
        ev3.LCX(0)  # BRAKE
    ])
    myEV3.send_direct_cmd(ops)


if __name__ == "__main__":
    move(100, 0)
    time.sleep(1)
    move(0, 0)

