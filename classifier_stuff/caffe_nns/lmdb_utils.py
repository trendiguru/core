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
from trendi.utils import imutils

#shellscript for mean comp:
#TOOLS=/home/ubuntu/repositories/caffe/build/tools
#DATA=/home/ubuntu/AdienceFaces/lmdb/Test_fold_is_0/gender_train_lmdb
#OUT=/home/ubuntu/AdienceFaces/mean_image/Test_folder_is_0

#$TOOLS/compute_image_mean.bin $DATA $OUT/mean.binaryproto

#get_ipython().system(u'data/mnist/get_mnist.sh')
#get_ipython().system(u'examples/mnist/create_mnist.sh')



#label = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_label', transform_param=dict(scale=1./255), ntop=1)
#data = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_data', transform_param=dict(scale=1./255), ntop=1)

################LMDB FUN (originally) RIPPED FROM http://deepdish.io/2015/04/28/creating-lmdb-in-python/
#############changes by awesome d.j. jazzy jer  awesomest hAckz0r evarr
def dir_of_dirs_to_lmdb(dbname,dir_of_dirs,test_or_train=None,max_images_per_class = 1000,resize_x=128,resize_y=128,avg_B=None,avg_G=None,avg_R=None,resize_w_bb=True,use_visual_output=False):
    print('writing to lmdb {} test/train {} max {} new_x {} new_y {} avgB {} avg G {} avgR {}'.format(dbname,test_or_train,max_images_per_class,resize_x,resize_y,avg_B,avg_G,avg_R))
    initial_only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    initial_only_dirs.sort()
 #   print(str(len(initial_only_dirs))+' dirs:'+str(initial_only_dirs)+' in '+dir_of_dirs)
    # txn is a Transaction object
    only_dirs = []
    for a_dir in initial_only_dirs:
        if (not test_or_train) or a_dir[0:4]==test_or_train[0:4]:
            #open and close db every class to cut down on memory
            #maybe this is irrelevant and we can do this once
            only_dirs.append(a_dir)
    print(str(len(only_dirs))+' relevant dirs:'+str(only_dirs)+' in '+dir_of_dirs)


    map_size = 1e13  #size of db in bytes, can also be done by 10X actual size  as in:
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
#    map_size = X.nbytes * 10

    if test_or_train:
        dbname = dbname+'.'+test_or_train
    print('writing to db:'+dbname)
    classno = 0
    image_number =0

    # txn is a Transaction object
    for a_dir in only_dirs:
        # do only test or train dirs if this param was sent
        image_number_in_class = 0
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
                cropped_name= os.path.join(fulldir,'cropped_'+a_file)
                #img_arr = mpimg.imread(fullname)  #if you don't have cv2 handy use matplotlib
                img_arr = cv2.imread(fullname)
                if img_arr is not None:
                    h_orig=img_arr.shape[0]
                    w_orig=img_arr.shape[1]
                    if(resize_x is not None):
#                            img_arr = imutils.resize_and_crop_image(img_arr, output_side_length = resize_x)
                        img_arr = imutils.resize_and_crop_image_using_bb(fullname, output_file=a_file,output_w=128,output_h=128,use_visual_output=use_visual_output)
                    h=img_arr.shape[0]
                    w=img_arr.shape[1]
                    print('img {} after resize w:{} h:{} (before was {}x{} name:{}'.format(image_number, h,w,h_orig,w_orig,fullname))
                    #    N = 1000
                    #    # Let's pretend this is interesting data
                    #    X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
                     #   y = np.zeros(N, dtype=np.int64)
                    if use_visual_output is True:
                        cv2.imshow('img',img_arr)
                        cv2.waitKey(0)
                    if avg_B is not None and avg_G is not None and avg_R is not None:
                        img_arr[:,:,0] = img_arr[:,:,0]-avg_B
                        img_arr[:,:,1] = img_arr[:,:,1]-avg_G
                        img_arr[:,:,2] = img_arr[:,:,2]-avg_R
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.channels = img_arr.shape[2]
                    datum.height = img_arr.shape[0]
                    datum.width = img_arr.shape[1]
                    datum.data = img_arr.tobytes()  # or .tostring() if numpy < 1.9
                    datum.label = classno
                    str_id = '{:08}'.format(image_number)
                    print('strid:{} w:{} h:{}'.format(str_id,datum.width,datum.height))
                    # The encode is only essential in Python 3
                    try:
                        txn.put(str_id.encode('ascii'), datum.SerializeToString())
            #            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
                        image_number += 1
                        image_number_in_class += 1
                    except:
                        e = sys.exc_info()[0]
                        print('some problem with lmdb:'+str(e))
                else:
                    print('couldnt read '+a_file)
        print('{} items in class {}'.format(image_number_in_class,classno))
        classno += 1
        env.close()


    #You can also open up and inspect an existing LMDB database from Python:
# assuming here that dataum.data, datum.channels, datum.width etc all exist as in dir_of_dirs_to_lmdb
def inspect_db(dbname):
    env = lmdb.open(dbname, readonly=True)
    with env.begin() as txn:
        n=0
        while(1):
            try:
                str_id = '{:08}'.format(n)
                print('strid:{} '.format(str_id))
             # The encode is only essential in Python 3
             #   txn.put(str_id.encode('ascii'), datum.SerializeToString())
                raw_datum = txn.get(str_id.encode('ascii'))
#                raw_datum = txn.get(b'00000000')
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(raw_datum)
                flat_x = np.fromstring(datum.data, dtype=np.uint8)
                x = flat_x.reshape(datum.channels, datum.height, datum.width)
                y = datum.label
 #               print('datum:'+str(datum))
                print('data {} y{} width {} height {} chan {}'.format(x,y,datum.width,datum.height,datum.channels))
                #Iterating <key, value> pairs is also easy:
                raw_input('enter to continue (n={})'.format(n))
                n+=1
            except:
                print('error getting record {} from db'.format(n))
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
    dir_of_dirs = '/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/dataset'
    dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/dataset'
    print('dir:'+dir_of_dirs)
#    h,w,d,B,G,R,n = imutils.image_stats_from_dir_of_dirs(dir_of_dirs)
    resize_x = 128
    #resize_y = int(h*128/w)
    resize_y=128
   # B=int(B)
   # G=int(G)
    #R=int(R)
    B=142
    G=151
    R=162
    dir_of_dirs_to_lmdb('testdb',dir_of_dirs,max_images_per_class =1000,test_or_train='test',resize_x=resize_x,resize_y=resize_y,avg_B=B,avg_G=G,avg_R=R)

#  weighted averages of 16 directories: h:1742.51040222 w1337.66435506 d3.0 B 142.492848614 G 151.617458606 R 162.580921717 totfiles 1442


#    dir_of_dirs_to_lmdb('testdb',dir_of_dirs,test_or_train='test',resize_x=128,resize_y=90,avg_B=101,avg_G=105,avg_R=123)
 #   inspect_db('testdb.test')

#    test_or_training_textfile(dir_of_dirs,test_or_train='train')
#    Utils.remove_duplicate_files('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1/')
#    image_stats_from_dir('/home/jr/