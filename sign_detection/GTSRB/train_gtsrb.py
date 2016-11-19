import caffe


def train(solver_name="lenet_solver.prototxt", gpu=False):
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # make sure, that the folder "data/gtsrb" is existing and writable
    solver = caffe.get_solver(solver_name)
    solver.solve()

    accuracy = solver.test_nets[0].blobs['accuracy'].data

    print("Accuracy: {:.3f}".format(accuracy))
