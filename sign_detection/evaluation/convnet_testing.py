#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import caffe
import argparse
import sign_detection.GTSRB.use_net as use_net

import itertools
import numpy as np
import matplotlib.pyplot as plt

TRAIN_DATA_ROOT = 'C:/development/GTSRB/Final_Test/Images/'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    original = cm[:]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for ix, row in enumerate(cm):
            cm[ix] = row / row.sum()
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, original[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--labelfile', type=str, required=True)
    args = parser.parse_args()

    caffe.set_mode_gpu()

    count = 0
    correct = 0
    matrix = np.zeros((43, 43), dtype=np.int)
    labels_set = set()

    net, transformer = use_net.load_net(args.proto, args.model)

    f = open(args.labelfile, "r")
    for line in f.readlines():
        parts = line.split()
        example_image = parts[0]
        label = int(parts[1])
        net = use_net.load_image(TRAIN_DATA_ROOT + example_image, net, transformer)

        category, probability = use_net.compute(net, "softmax")
        plabel = category
        count += 1
        iscorrect = label == plabel
        correct += (1 if iscorrect else 0)
        matrix[label][plabel] += 1
        labels_set.update([label, plabel])
        if not iscorrect:
            print("\rError: expected %i but predicted %i" \
                  % (label, plabel))

        sys.stdout.write("\rAccuracy: %.1f%%" % (100. * correct / count))
        sys.stdout.flush()

    print(", %i/%i corrects" % (correct, count))

    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            sys.stdout.write('.')
            print "(%i , %i) | %i" % (l, pl, matrix[l][pl])

    for l in labels_set:
        sys.stdout.write(str(l))
        sys.stdout.write(";")
        for pl in labels_set:
            sys.stdout.write(str(matrix[l][pl]))
            sys.stdout.write(";")
        print ""

    plot_confusion_matrix(matrix, labels_set, normalize=False, title='Confusion matrix, without normalization')

    plt.figure()
    plot_confusion_matrix(matrix, classes=labels_set, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
