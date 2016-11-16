import timeit
import caffe


def train(solver_name, gpu=True):
    if (gpu):
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    solver = caffe.get_solver(solver_name)
    #    solver.solve()

    print "Starting solving"
    for itt in range(6):
        print "Step " + str(itt)
        solver.step(100)
        print 'itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 50))

    accuracy = check_accuracy(solver.test_nets[0], 50)

    print("Accuracy: {:.3f}".format(accuracy))


def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))


def check_accuracy(net, num_batches, batch_size=128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests):  # for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


start = timeit.default_timer()

train("solver.prototxt", True)

stop = timeit.default_timer()
print("Start: " + str(start))
print("Time:  " + str(stop - start))
