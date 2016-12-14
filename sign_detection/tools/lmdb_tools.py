from random import shuffle
from sys import getsizeof

import lmdb
import caffe
from scipy.misc import imshow

import sign_detection.GTSDB.SlidingWindow.batchloader
import numpy as np


def create_sliding_window_lmdb_from_gtsdb(name, gtsdb_root, window_size, number_entries, overlap):
    params = {'gtsdb_root': gtsdb_root, 'window_size': window_size}
    loader = sign_detection.GTSDB.SlidingWindow.batchloader.BatchLoader(params, num=100000, fraction=0.5)
    # load one image to find a good size
    regions = collect_regions(gtsdb_root, window_size, 5000, overlap, loader)

    image, label = regions[0]
    datum = caffe.io.array_to_datum(image.astype(np.uint8), int(label))

    map_size = int((getsizeof(datum.SerializeToString())) * number_entries * 10)
    print "Database size: " + str(map_size / 2 ** 20) + "MB"

    env = lmdb.open(name, map_size=map_size)
    with env.begin(write=True) as txn:
        for i in range(int(number_entries * 0.8)):
            while len(regions) < 1:
                regions = collect_regions(gtsdb_root, window_size, 1000, overlap, loader)
            image, label = regions.pop()
            # txn is a Transaction object
            datum = caffe.io.array_to_datum(image.astype(np.uint8), int(label))
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

            if i % 1000 == 0: print i

    env = lmdb.open(name + "_test", map_size=map_size)
    with env.begin(write=True) as txn:
        for i in range(int(number_entries * 0.2)):
            while len(regions) < 1:
                regions = collect_regions(gtsdb_root, window_size, 1000, overlap, loader)
            image, label = regions.pop()
            # txn is a Transaction object
            datum = caffe.io.array_to_datum(image.astype(np.uint8), int(label))
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

            if i % 1000 == 0: print i


def collect_regions(gtsdb_root, window_size, number_entries, overlap, loader):
    # load one image to find a good size
    regions = loader.generate_regions_from_image(overlap=overlap)
    while len(regions) < 1:
        regions = loader.generate_regions_from_image(overlap=overlap)

    result = []
    for i in range(number_entries):
        while len(regions) < 1:
            regions = loader.generate_regions_from_image(overlap=overlap)
        result.append(regions.pop())

    shuffle(result)
    return result

def get_image_from_lmdb(lmdb_name, image_name):
    env = lmdb.open(lmdb_name, readonly=True)
    with env.begin() as txn:
        raw_datum = txn.get(image_name)

    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    x = caffe.io.datum_to_array(datum)
    y = datum.label
    return x, y


def iterate_lmdb(lmdb_name):
    env = lmdb.open(lmdb_name, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)

            x = caffe.io.datum_to_array(datum)
            print datum.label
            imshow(x)

# create_sliding_window_lmdb_from_gtsdb("gtsdb_sliding_window", "/home/philipp/development/FullIJCNN2013", 64, 13000, 0.4)
# create_sliding_window_lmdb_from_gtsdb("gtsdb_sliding_window_test", "/home/philipp/development/FullIJCNN2013", 64, 20000, 0.4)
# iterate_lmdb("test")
