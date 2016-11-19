__author__ = 'jeremy'

import numpy as np
import cv2
import os 
import json
import sys
import time

from trendi.paperdoll import pd_falcon_client
from trendi.utils import imutils
from trendi import Utils,constants

def get_pd_results(url=None,filename=None):
    if url is not None:
        print('getting pd results for '+url)
        image = Utils.get_cv2_img_array(url)
    elif filename is not None:
        print('getting pd results for '+filename)
        image = cv2.imread(filename)
    if image is None:
        print('image came back none')
        return None
    seg_res = pd_falcon_client.pd(image)
#    print('segres:'+str(seg_res))
    if filename is not None:
        imgfilename = os.path.basename(filename) #use the actual on-disk filename if thats what we started with, otherwise use the name generated by pd
        #this will et saved to /home/jeremy/pd_output/filename
    else:
        try:
            imgfilename = seg_res['filename']+'.jpg'
        except:
            print('some error on imgfile name')
            imgfilename = str(int(time.time()))+'.jpg'
    print('filename '+imgfilename)
    if not 'mask' in seg_res:
        print('pd did not return mask')
        print seg_res
        return
    mask = seg_res['mask']
    label_dict = seg_res['label_dict']
    print('labels:'+str(label_dict))
    pose = seg_res['pose']
    mask_np = np.array(mask, dtype=np.uint8)
    print('masksize '+str(mask_np.shape))
    pose_np = np.array(pose, dtype=np.uint8)
    print('posesize '+str(pose_np.shape))
#    print('returned url '+seg_res['url'])
    convert_and_save_results(mask_np, label_dict, pose_np, imgfilename, image, url)
    dir = constants.pd_output_savedir
    Utils.ensure_dir(dir)
    full_name = os.path.join(dir,imgfilename)
#            full_name = filename
    bmp_name = full_name.strip('.jpg') + ('.bmp')  #this bmp should get generated by convert_and_save_results
    imutils.show_mask_with_labels(bmp_name,constants.fashionista_categories_augmented_zero_based,original_image = full_name,save_images=True)
#these are 1-based not 0-based

def convert_and_save_results(mask, label_names, pose,filename,img,url,forwebtool=True):
    '''
    This saves the mask using the labelling fashionista_categories_augmented_zero_based
    :param mask:
    :param label_names:
    :param pose:
    :param filename:
    :param img:
    :param url:
    :return:
     '''
    fashionista_ordered_categories = constants.fashionista_categories_augmented_zero_based  #constants.fashionista_categories
    new_mask=np.ones(mask.shape)*255  # anything left with 255 wasn't dealt with
    success = True #assume innocence until proven guilty
    print('attempting convert and save')
    for label in label_names: # need these in order
        if label in fashionista_ordered_categories:
            fashionista_index = fashionista_ordered_categories.index(label) + 0  # number by  0=null, 55=skin  , not 1=null,56=skin
            pd_index = label_names[label]
            pixlevel_v2_index = constants.fashionista_aug_zerobased_to_pixlevel_categories_v2[pd_index]
            if pixlevel_v2_index is None:
                pixlevel_v2_index = 0
#            new_mask[mask==pd_index] = fashionista_index
            print('old index '+str(pd_index)+' for '+str(label)+': gets new index:'+str(fashionista_index)+' and newer index '+str(pixlevel_v2_index)+fashionista_ordered_categories[fashionista_index]+':'+constants.pixlevel_categories_v2[pixlevel_v2_index])
            new_mask[mask==pd_index] = pixlevel_v2_index
        else:
            print('label '+str(label)+' not found in regular cats')
            success=False
    if 255 in new_mask:
        print('didnt fully convert mask')
        success = False
    if success:
        try:
            dir = constants.pd_output_savedir
            Utils.ensure_dir(dir)
            full_name = os.path.join(dir,filename)
#            full_name = filename
            print('writing output img to '+str(full_name))
            cv2.imwrite(full_name,img)
            bmp_name = full_name.strip('.jpg') + ('_pixv2.bmp')
            if forwebtool:
                new_mask[:,:,0:2]=0 #zero out the B,G for webtool - leave only R
                bmp_name=bmp_name.replace('.bmp','_webtool.bmp')
            print('writing output bmp to '+str(bmp_name))
            cv2.imwrite(bmp_name,new_mask)
            pose_name = full_name.strip('.jpg')+'.pose'
#            print('orig pose '+str(pose))
#            print('writing pose to '+str(pose_name))
            with open(pose_name, "w+") as outfile:
                print('succesful open, attempting to write pose')
                poselist=pose[0].tolist()
#                json.dump([1,2,3], outfile, indent=4)
                json.dump(poselist,outfile, indent=4)
            if url is not None:
                url_name = full_name.strip('.jpg')+'.url'
                print('writing url to '+str(url_name))
                with open(url_name, "w+") as outfile2:
                    print('succesful open, attempting to write:'+str(url))
                    outfile2.write(url)
            return
        except:
            print('fail in convert_and_save_results dude, bummer')
            print(str(sys.exc_info()[0]))
            return
    else:
        print('didnt fully convert mask, or unkown label in convert_and_save_results')
        success = False
        return


if __name__ == "__main__":
 #   url = 'https://s-media-cache-ak0.pinimg.com/736x/3a/85/79/3a857905d8814faf49910f9c2b9806a8.jpg'
 #   get_pd_results(url=url)
    dir = '/home/jeremy/image_dbs/tamara_berg_street_to_shop/photos'
    tbphotos = [os.path.join(dir,im) for im in os.listdir(dir)]
    n = len(tbphotos)
    start = 0
    if len(sys.argv)>1:
        start = int(sys.argv[1])*n/3
    if len(sys.argv)>2:
        offset = int(sys.argv[2])

    print(str(len(tbphotos))+' images in '+dir+', starting at '+str(start))
    for f in tbphotos[start:]:
        i = tbphotos.index(f)
        print('working on image {} '.format(i))
        get_pd_results(filename=f)
        time.sleep(1)
