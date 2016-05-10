__author__ = 'jeremy'
from trendi.paperdoll import paperdoll_parse_enqueue
import time
import numpy as np
import cv2
import os
from __future__ import print_function
import operator

from trendi.utils import imutils
urls=[]
dts=[]
#urls.append('http://notapicture.jpg')

filenames =  []
filenames.append( '/home/netanel/meta/dataset/test1/product_9415_photo_3295_bbox_336_195_339_527.jpg')
dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test'
filenames = [os.path.join(dir,f) for f in os.listdir(dir) if '.jpg' in f]


urls.append('http://i.imgur.com/ahFOgkm.jpg')
urls.append('http://40.media.tumblr.com/b81282b59ab467eab299801875bc3670/tumblr_mhc692qtpq1r647c2o1_500.jpg')
urls.append('http://www.hollywoodtuna.com/images/christina_hendricks_green_small.jpg')
urls.append('http://aws-cdn.dappered.com/wp-content/uploads/2014/03/CH2008.jpg')
urls.append('http://www.fashiontrendspk.com/wp-content/uploads/Emma-Stone-in-Lanvin-at-the-2012-Golden-Globe-Awards-450x345.jpg')
urls.append('http://gingertalk.com/wp-content/uploads/2013/12/golddress-200x300.jpg')
urls.append('https://s-media-cache-ak0.pinimg.com/736x/fb/92/50/fb9250e68e63f6862da24bfb3ae17b0a.jpg')
urls.append('https://s-media-cache-ak0.pinimg.com/736x/71/82/42/7182428f1c4b584fe084823791aa9d59.jpg')
urls.append('https://s-media-cache-ak0.pinimg.com/736x/c1/a4/56/c1a456661a699c99e9a019648ba928a2.jpg')
urls.append('http://gingerparrot.co.uk/wp/wp-content/uploads/2014/04/Katy-B-Red-Hair-White-Clothes-Still-Video.jpg')
urls.append('http://media2.popsugar-assets.com/files/2010/08/34/5/192/1922153/9621f2d8749ddac7_red-main/i/What-Kind-Makeup-Wear-Youre-Redhead-Wearing-Red-Dress.jpg')


for f in filenames:
    print('sending img '+f)
    im = cv2.imread(f)
    start_time = time.time()
    retval = paperdoll_parse_enqueue.paperdoll_enqueue(im)
    print('waiting',end='')
    while not retval.is_finished:
        time.sleep(1)
        print('.', end="")
    mask, labels = retval.result[:2]
    end_time = time.time()
    dt=end_time-start_time
    dts.append(dt)
    parse_name = f.split('.jpg')[0]+'_parse.png'
    cv2.imwrite(parse_name,mask)
    labeloutname = f.split('.jpg')[0]+'_labels.txt'
    print('labels:'+str(labels))
    sorted_labels=sorted(labels.items(),key=operator.itemgetter(1))
    print('sorted labels :'+str(sorted_labels))
    labs_only = [i[0] for i in sorted()]
    print('labsonly '+str(labs_only))

    imutils.show_mask_with_labels(parse_name,labs_only,save_images=True)

    with open(labeloutname,'w') as labelfile:
        labelfile.write(labels)
    if retval is not None:
        print('retval:' + str(retval.result)+' time:'+str(dt))
    else:
        print('no return val (None)')

if(0):
    for f in filenames:
        print('sending filenames')
        start_time = time.time()
        retval = paperdoll_parse_enqueue.paperdoll_enqueue(f, async=False,use_parfor=False)  #True,queue_name='pd_parfor')
        end_time = time.time()
        dt=end_time-start_time
        dts.append(dt)
        if retval is not None:
            print('retval:' + str(retval.result)+' time:'+str(dt))
        else:
            print('no return val (None)')



    for url in urls:
        start_time = time.time()
        retval = paperdoll_parse_enqueue.paperdoll_enqueue(url, async=False,use_parfor=False)  #True,queue_name='pd_parfor')
        end_time = time.time()
        dt=end_time-start_time
        dts.append(dt)
        if retval is not None:
            print('retval:' + str(retval.result)+' time:'+str(dt))
        else:
            print('no return val (None)')

    means=np.mean(dts)
    std=np.std(dts)
    print('mean:' + str(means)+' std:'+str(std))