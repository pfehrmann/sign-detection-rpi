import sign_detection.EV3.EV3 as ev3
import struct


def get_folder(dir, my_ev3):
    directory = dir
    ops = b''.join([
        ev3.opFile,
        ev3.GET_FOLDERS,
        ev3.LCS(directory),
        ev3.GVX(0)
    ])
    reply = my_ev3.send_direct_cmd(ops, global_mem=1)
    num = struct.unpack('<B', reply[5:])[0]
    print(
        "Directory '{}' has {} subdirectories".format(
            directory,
            num
        )
    )

if __name__ == "__main__":
    myEv3 = ev3.EV3(protocol=ev3.USB, host='00:16:53:47:92:46')
    show_image("../prjs/Signs/SIGN_SPEED_30.rgf", myEv3)
