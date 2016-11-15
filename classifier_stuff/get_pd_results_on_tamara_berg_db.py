__author__ = 'jeremy'

import numpy as np
import cv2
import os 
import json
import sys

from trendi.paperdoll import pd_falcon_client
from trendi.utils import imutils
from trendi import Utils,constants

def get_pd_results(url):
    print('getting pd results for '+url)
    image = Utils.get_cv2_img_array(url)
    if image is None:
        print('image came back none')
    seg_res = pd_falcon_client.pd(image)
#    print('segres:'+str(seg_res))
    imgfilename = seg_res['filename']
    print('filename '+imgfilename)
    cv2.imwrite(imgfilename,image)

    mask = seg_res['mask']
    label_dict = seg_res['label_dict']
    pose = seg_res['pose']
    mask_np = np.array(mask, dtype=np.uint8)
    print('masksize '+mask_np.shape)
    pose_np = np.array(pose, dtype=np.uint8)
    print('posesize '+pose_np.shape)
#    print('returned url '+seg_res['url'])
    convert_and_save_results(mask_np, label_dict, pose_np, imgfilename, image, url)
    maskfilename = 'testout.png'
    cv2.imwrite(maskfilename,seg_res)
    imutils.show_mask_with_labels(maskfilename,constants.fashionista_categories_augmented_zero_based,original_image = imgfilename,save_images=True)

def convert_and_save_results(mask, label_names, pose,filename,img,url):
    fashionista_ordered_categories = constants.fashionista_categories
        #in case it changes in future - as of 2/16 this list goes a little something like this:
        #fashionista_categories = ['null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse','boots',
          #                'blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings','scarf','hat',
            #              'top','cardigan','accessories','vest','sunglasses','belt','socks','glasses','intimate',
              #            'stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges','ring',
                #          'flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch','pumps','wallet',
                  #        'bodysuit','loafers','hair','skin']
    new_mask=np.ones(mask.shape)*255  # anything left with 255 wasn't dealt with
    success = True #assume innocence until proven guilty
    print('attempting convert and save')
    for label in label_names: # need these in order
        if label in fashionista_ordered_categories:
            fashionista_index = fashionista_ordered_categories.index(label) + 1  # start w. 1=null,56=skin
            pd_index = label_names[label]
       #     print('old index '+str(pd_index)+' for '+str(label)+': gets new index:'+str(fashionista_index))
            new_mask[mask==pd_index] = fashionista_index
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
            bmp_name = full_name.strip('.jpg') + ('.bmp')
            print('writing output img to '+str(full_name))
            cv2.imwrite(full_name,img)
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
    url = 'https://s-media-cache-ak0.pinimg.com/736x/3a/85/79/3a857905d8814faf49910f9c2b9806a8.jpg'
    get_pd_results(url)