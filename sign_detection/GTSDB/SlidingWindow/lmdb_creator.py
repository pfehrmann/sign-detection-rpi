from sys import getsizeof

import lmdb
import caffe
import batchloader


def create_lmdb(name, gtsdb_root, window_size, number_entries):
    params = {'gtsdb_root': gtsdb_root, 'window_size': window_size}
    loader = batchloader.BatchLoader(params, num=100000, fraction=0.5)

    image, label = loader.next_window()
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = 3
    datum.height = window_size
    datum.width = window_size
    datum.data = image.tobytes()  # or .tostring() if numpy < 1.9
    datum.label = int(label)
    str_id = '{:08}'.format(0)

    loader.next_window()
    map_size = int((getsizeof(datum.SerializeToString()) + 1) * number_entries * 10)
    print "Database size: " + str(map_size / 2 ** 30) + "GB"

    env = lmdb.open(name, map_size=map_size)
    with env.begin(write=True) as txn:
        for i in range(number_entries):
            image, label = loader.next_window()
            # txn is a Transaction object
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 3
            datum.height = window_size
            datum.width = window_size
            datum.data = image.tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(label)
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

            if i % 500 == 0: print i


create_lmdb("test", "/home/philipp/development/FullIJCNN2013", 64, 200000)
