from sys import getsizeof

import lmdb
import caffe
import sign_detection.GTSDB.SlidingWindow.batchloader
import numpy as np


def create_sliding_window_lmdb_from_gtsdb(name, gtsdb_root, window_size, number_entries):
    params = {'gtsdb_root': gtsdb_root, 'window_size': window_size}
    loader = sign_detection.GTSDB.SlidingWindow.batchloader.BatchLoader(params, num=100000, fraction=0.5)

    image, label = loader.next_window()
    datum = caffe.io.array_to_datum(image.astype(np.uint8), label)
    str_id = '{:08}'.format(0)

    loader.next_window()
    map_size = int((getsizeof(datum.SerializeToString())) * number_entries * 10)
    print "Database size: " + str(map_size / 2 ** 20) + "MB"

    env = lmdb.open(name, map_size=map_size)
    for a in range(number_entries / 1000):
        if a % 5 == 0: print a * 1000
        with env.begin(write=True) as txn:
            for i in range(1000):
                image, label = loader.next_window()
                # txn is a Transaction object
                datum = caffe.io.array_to_datum(image.astype(np.uint8), label)
                str_id = '{:08}'.format(i)

                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())


def get_image_from_lmdb(lmdb_name, image_name):
    env = lmdb.open(lmdb_name, readonly=True)
    with env.begin() as txn:
        raw_datum = txn.get(image_name)

    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    x = caffe.io.datum_to_array(datum)
    y = datum.label
    return x, y


create_sliding_window_lmdb_from_gtsdb("gtsdb_sliding_dataset", "/home/philipp/development/FullIJCNN2013", 64, 100000)
