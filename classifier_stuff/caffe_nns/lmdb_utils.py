# coding: utf-8
__author__ = 'jeremy'
from pylab import *
import caffe
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import lmdb
from PIL import Image

#get_ipython().system(u'data/mnist/get_mnist.sh')
#get_ipython().system(u'examples/mnist/create_mnist.sh')



#label = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_label', transform_param=dict(scale=1./255), ntop=1)
#data = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_data', transform_param=dict(scale=1./255), ntop=1)

################LMDB FUN (originally) RIPPED FROM http://deepdish.io/2015/04/28/creating-lmdb-in-python/
#############changes by awesome d.j. jazzy jer  awesomest hAckz0r evarr
def dir_of_dirs_to_lmdb(dbname,dir_of_dirs,test_or_train=None,max_images_per_class = 100):
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    only_dirs.sort()
    print(str(len(only_dirs))+' dirs:'+str(only_dirs)+' in '+dir_of_dirs)
    map_size = 1e13  #size of db in bytes, can also be done by 10X actual size  as in:
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
#    map_size = X.nbytes * 10

    if test_or_train:
        dbname = dbname+'.'+test_or_train
    classno = 0
    image_number =0

    # txn is a Transaction object
    for a_dir in only_dirs:
        # do only test or train dirs if this param was sent
        image_number_in_class = 0
        if (not test_or_train) or dir[0:4]==test_or_train[0:4]:
            #open and close db every class to cut down on memory
            #maybe this is irrelevant and we can do this once
            env = lmdb.open(dbname, map_size=map_size)
            with env.begin(write=True) as txn:
                fulldir = os.path.join(dir_of_dirs,a_dir)
                print('fulldir:'+str(fulldir))
                only_files = [f for f in os.listdir(fulldir) if os.path.isfile(os.path.join(fulldir, f))]
                n = len(only_files)
                print('n files {} in {}'.format(n,dir))
                for n in range(0,min(max_images_per_class,len(only_files))):
                    a_file =only_files[n]
                    fullname = os.path.join(fulldir,a_file)
                    #img_arr = mpimg.imread(fullname)  #if you don't have cv2 handy use matplotlib
                    img_arr = cv2.imread(fullname)
                    if img_arr is not None:
                        #    N = 1000
                        #    # Let's pretend this is interesting data
                        #    X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
                         #   y = np.zeros(N, dtype=np.int64)
                    #    cv2.imshow('img',img_arr)
                     #   cv2.waitKey(10)
                        datum = caffe.proto.caffe_pb2.Datum()
                        datum.channels = img_arr.shape[2]
                        datum.height = img_arr.shape[0]
                        datum.width = img_arr.shape[1]
                        datum.data = img_arr.tobytes()  # or .tostring() if numpy < 1.9
                        datum.label = classno
                        str_id = '{:08}'.format(image_number)
                        #print('strid:'+str(str_id))
                        # The encode is only essential in Python 3
                        txn.put(str_id.encode('ascii'), datum.SerializeToString())
            #            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
                        image_number += 1
                        image_number_in_class += 1
                    else:
                        print('couldnt read '+a_file)
            print('{} items in class {}'.format(image_number_in_class,classno))
            classno += 1
            env.close()





    #You can also open up and inspect an existing LMDB database from Python:
def inspect_db(dbname):
    env = lmdb.open(dbname, readonly=True)
    with env.begin() as txn:
        try:
            raw_datum = txn.get(b'00000000')
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = np.fromstring(datum.data, dtype=np.uint8)
            x = flat_x.reshape(datum.channels, datum.height, datum.width)
            y = datum.label
            #Iterating <key, value> pairs is also easy:
            raw_input('enter to continue')

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            print(key, value)


def crude_lmdb():
    in_db = lmdb.open('image-lmdb', map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(inputs):
            # load image:
            # - as np.uint8 {0, ..., 255}
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)
 #           im = np.array(Image.open(in_)) # or load whatever ndarray you need
  #          im = im[:,:,::-1]
   #         im = im.transpose((2,0,1))
    #        im_dat = caffe.io.array_to_datum(im)
      #      in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
            im = np.array(Image.open(in_)) # or load whatever ndarray you need
            im = im[:,:,::-1]
            im = im.transpose((2,0,1))
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    in_db.close()

if __name__ == "__main__":
    dir_of_dirs = '/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train'
    print('dir:'+dir_of_dirs)
    dir_of_dirs_to_lmdb('testdb',dir_of_dirs,test_or_train='test')
    inspect_db('testdb.test')

#    test_or_training_textfile(dir_of_dirs,test_or_train='train')
#    Utils.remove_duplicate_files('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1/')
#    image_stats_from_dir('/home/jr/