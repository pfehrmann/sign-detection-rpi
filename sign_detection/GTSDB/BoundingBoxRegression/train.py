import caffe

import sign_detection.tools.batchloader as bl
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un


def train(solver_name="solver.prototxt", gpu=False):
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # Create solver
    solver = caffe.get_solver(solver_name)

    solver.solve()





train("net_separate/solver.prototxt", True)
