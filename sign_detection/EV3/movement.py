import sign_detection.EV3.EV3 as ev3
import time

def move(speed, turn, my_e_v3):
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
    my_e_v3.send_direct_cmd(ops)


def stop(my_e_v3):
    ops = b''.join([
        ev3.opOutput_Stop,
        ev3.LCX(0),  # LAYER
        ev3.LCX(ev3.PORT_B + ev3.PORT_C),  # NOS
        ev3.LCX(0)  # BRAKE
    ])
    my_e_v3.send_direct_cmd(ops)


if __name__ == "__main__":
    myEv3 = ev3.EV3(protocol=ev3.USB, host='00:16:53:47:92:46')
    move(10, 0, myEv3)
    time.sleep(1)
    move(0, 0, myEv3)
    time.sleep(0.1)
    stop(myEv3)
