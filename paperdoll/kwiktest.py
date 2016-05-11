from __future__ import print_function
__author__ = 'jeremy'
from trendi.paperdoll import paperdoll_parse_enqueue
import time
import numpy as np
import cv2
import os
import operator
import sys

from trendi.utils import imutils
from trendi import pipeline

urls=[]
dts=[]
#urls.append('http://notapicture.jpg')

filenames =  []
filenames.append( '/home/netanel/meta/dataset/test1/product_9415_photo_3295_bbox_336_195_339_527.jpg')
#filenames=filenames[100:]

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



def get_pd_masks_for_dir(indir,outdir):
    filenames = [f for f in os.listdir(indir) if '.jpg' in f]
    print('found {} files in {}'.format(len(filenames),indir))
    for f in filenames:
        print('sending img '+f)
        full_imgname = os.path.join(indir,f)
        im = cv2.imread(full_imgname)
        start_time = time.time()
        retval = paperdoll_parse_enqueue.paperdoll_enqueue(im)
        print('waiting',end='')
        while not retval.is_finished:
            time.sleep(1)
            print('.', end="")
            sys.stdout.flush()
        mask, labels = retval.result[:2]
        end_time = time.time()
        dt=end_time-start_time
        dts.append(dt)
        parse_name = f.split('.jpg')[0]+'_pdparse.bmp'
        full_parse_name = os.path.join(outdir,parse_name)
        cv2.imwrite(full_parse_name,mask)
        print('labels:'+str(labels))
        sorted_labels=sorted(labels.items(),key=operator.itemgetter(1))
        print('sorted labels :'+str(sorted_labels))
        labs_only = [i[0] for i in sorted_labels]
        print('labsonly '+str(labs_only))
        imutils.show_mask_with_labels(full_parse_name,labs_only,save_images=True)

        aftermask = pipeline.after_pd_conclusions(mask, labels, face=None)
        after_pd_conclusions_name = full_parse_name.split('_pdparse.bmp')[0]+'_pdconclusions.bmp'
        cv2.imwrite(after_pd_conclusions_name,aftermask)
        imutils.show_mask_with_labels(after_pd_conclusions_name,labs_only,save_images=True)

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

if __name__ =="__main__":
    indir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test'
    outdir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/pd'
    get_pd_masks_for_dir(indir,outdir)