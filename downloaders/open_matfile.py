__author__ = 'jeremy'

import scipy.io as sio
import cv2
import os
import numpy as np
import struct

def load_malldataset(filename='/data/jeremy/image_dbs/hls/mall_dataset/mall_gt.mat'):
    mat = sio.loadmat(filename)
    for k,v in mat.iteritems():
        print(k)

#    print(mat['__globals__'])
#    print(mat['__header__'])
#    print(mat['__version__'])
    #print(mat['count'])
    #raw_input('ret to cont')
    #print(mat['frame'])

    return mat['frame']


def inspect_matlab_dataset(filename='/data/jeremy/image_dbs/hls/mall_dataset/mall_gt.mat'):
    mat = sio.loadmat(filename)
    for k,v in mat.iteritems():
        print(k)
    raw_input('return to continue')
    return mat

def read_seq(path):
    '''
    from https://gist.github.com/psycharo/7e6422a491d93e1e3219/
    :param path:
    :return:
    '''

    def read_header(ifile):
        feed = ifile.read(4)
        norpix = ifile.read(24)
        version = struct.unpack('@i', ifile.read(4))
        length = struct.unpack('@i', ifile.read(4))
        assert(length != 1024)
        descr = ifile.read(512)
        params = [struct.unpack('@i', ifile.read(4))[0] for i in range(0,9)]
        fps = struct.unpack('@d', ifile.read(8))
        # skipping the rest
        ifile.read(432)
        image_ext = {100:'raw', 102:'jpg',201:'jpg',1:'png',2:'png'}
        return {'w':params[0],'h':params[1],
                'bdepth':params[2],
                'ext':image_ext[params[5]],
                'format':params[5],
                'size':params[4],
                'true_size':params[8],
                'num_frames':params[6]}

    ifile = open(path, 'rb')
    params = read_header(ifile)
    bytes = open(path, 'rb').read()

    # this is freaking magic, but it works
    extra = 8
    s = 1024
    seek = [0]*(params['num_frames']+1)
    seek[0] = 1024

    images = []

    for i in range(0, params['num_frames']):
        tmp = struct.unpack_from('@I', bytes[s:s+4])[0]
        s = seek[i] + tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s+1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        seek[i+1] = s
        nbytes = struct.unpack_from('@i', bytes[s:s+4])[0]
        I = bytes[s+4:s+nbytes]

        tmp_file = '/tmp/img%d.jpg' % i
        open(tmp_file, 'wb+').write(I)

        img = cv2.imread(tmp_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

def show_rects(n,head_positions):
    name = '/data/jeremy/image_dbs/hls/mall_dataset/frames/seq_%06d'%n+'.jpg' #
    if not os.path.isfile(name):
        print('{} is not a file'.format(name))
        exit()
    print(name)
    img_arr = cv2.imread(name)
    xsize = 5
    im_height,im_width = img_arr.shape[0:2]
    for i in range(len(head_positions)):
        print head_positions[i]
        x=int(head_positions[i][0])
        y=int(head_positions[i][1])
#        y=im_height-y
        cv2.rectangle(img_arr,(x-xsize,y-xsize),(x+xsize,y+xsize),color=[255,0,100],thickness=2)
    cv2.imshow('img',img_arr)
    cv2.waitKey(0)
    return(img_arr)

if __name__=="__main__":
    f1 = inspect_matlab_dataset(filename='/data/jeremy/image_dbs/hls/mall_dataset/mall_feat.mat')
    frame = load_malldataset()
    print frame.shape
    n=1
    name = '/data/jeremy/image_dbs/hls/mall_dataset/frames/seq_%06d'%n+'.jpg'
    if not os.path.isfile(name):
        print('{} is not a file'.format(name))
        exit()
    print(name)
    img = cv2.imread(name)
    height , width , layers =  img.shape
    video = cv2.VideoWriter('video.xvid',-1,1,(width,height))


    for n in range(1,200):
        framedat = frame[0,n]
        head_outer = framedat[0,:][0]
        head_positions = head_outer[0]
        img=show_rects(n,head_positions)

        video.write(img)
        cv2.destroyAllWindows()
        video.release()