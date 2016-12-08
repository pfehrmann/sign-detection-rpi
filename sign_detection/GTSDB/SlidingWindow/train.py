import argparse
import timeit
import caffe


def train(solver_name, gpu=True, use_solver=False, iters=100):
    """
    Train a given solver using either the solver options or custom ones. Either the GPU or CPU can be used for training.
    :param solver_name: The name (and potentially also the path) of the solver file.
    :param gpu: Use the GPU?
    :param use_solver: Use all the settings from the solver?
    :param iters: The number of iterations. Per iteration 100 steps are performed. This setting is ignored if use_solver == True
    :return: None
    :returns: None

    :type solver_name: str
    :type gpu: bool
    :type use_solver: bool
    :type iters: int
    """

    # Set the correct device for solving
    if (gpu):
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # Initialize the solver
    solver = caffe.get_solver(solver_name)

    print "Starting solving"
    if use_solver:
        # Use the solver settings
        solver.solve()

    else:
        # Use the custom settings
        for itt in range(iters):
            print "Step " + str(itt)
            solver.step(100)

    # Save the model
    solver.net.save("model.caffemodel")


def parse_arguments():

    # Create the parser
    parser = argparse.ArgumentParser(description='Train a net to detect regions in images')
    parser.add_argument('solver', type=str, help='The solver file to use')
    parser.add_argument('-g', '--no-gpu', dest='gpu', action='store_false', help='Train without GPU?')
    parser.add_argument('-s', '--usesolver', type=bool, default=False, help='Use only the settings from the solver?')
    parser.add_argument('-i', '--iterations', type=int, default=6, help='The number of iterations (ignored, when using solvers settings)')
    parser.set_defaults(gpu=True)

    # Read the input arguments
    args = parser.parse_args()

    # start timing
    start = timeit.default_timer()

    train(args.solver, args.gpu, args.usesolver, args.iterations)

    # stop timing and print results
    stop = timeit.default_timer()
    print("Time:  " + str(stop - start))

parse_arguments()
#train("Mini_net_solver.prototxt", iters=6)
