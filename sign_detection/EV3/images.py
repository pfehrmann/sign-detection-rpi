import sign_detection.EV3.EV3 as ev3


def show_image(image_path, my_ev3):
    ops = b''.join([
        ev3.opUI_Draw,
        ev3.TOPLINE,
        ev3.LCX(0),  # ENABLE
        ev3.opUI_Draw,
        ev3.BMPFILE,
        ev3.LCX(1),  # COLOR
        ev3.LCX(0),  # X0
        ev3.LCX(0),  # Y0
        ev3.LCS(image_path),  # NAME
        ev3.opUI_Draw,
        ev3.UPDATE
    ])
    reply = my_ev3.send_direct_cmd(ops)

if __name__ == "__main__":
    myEv3 = ev3.EV3(protocol=ev3.USB, host='00:16:53:47:92:46')
    show_image("../prjs/Signs/SIGN_SPEED_30.rgf", myEv3)
