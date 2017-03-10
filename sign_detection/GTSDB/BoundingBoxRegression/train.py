import caffe

from sign_detection.tools.dataset_augemntation import create_directory


def train(solver_name="solver.prototxt", gpu=False):
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    create_directory("data/snapshot/bla")
    # Create solver
    solver = caffe.get_solver(solver_name)

    solver.solve()


train("net_separate/solver.prototxt", True)
