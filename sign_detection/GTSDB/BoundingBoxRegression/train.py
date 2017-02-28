import caffe


import sign_detection.tools.batchloader as bl
import sign_detection.GTSDB.ActivationMapBoundingBoxes.use_net as un


def train(solver_name="solver.prototxt", gpu=False):
    if gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # Load the net
    net = un.load_net("../ActivationMapBoundingBoxes/mini_net/deploy.prototxt",
                      "../ActivationMapBoundingBoxes/mini_net/weights.caffemodel")

    # setup the detector
    detector = un.Detector(net, minimum=0.9999, use_global_max=False, threshold_factor=0.75, draw_results=False,
                           zoom=[1], area_threshold_min=1200, area_threshold_max=50000,
                           activation_layer="activation",
                           out_layer="softmax", display_activation=False, blur_radius=1, size_factor=0.5,
                           faster_rcnn=True, modify_average_value=True, average_value=30)

    # Get images to train with
    images = bl.get_images_and_regions('/home/leifb/Development/Data/GTSDB', 0, 10)

    # Create solver
    solver = caffe.get_solver(solver_name)

    for img in images:
        img_raw = bl.load(img)

        rois, activation = detector.identify_regions(img_raw)

        print len(rois)

        solver.net.params['ip1_1'].dat = activation



    accuracy = solver.test_nets[0].blobs['accuracy'].data

    print("Accuracy: {:.3f}".format(accuracy))


train("solver.prototxt", True)
