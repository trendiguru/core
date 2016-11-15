__author__ = 'jeremy'

import numpy as np

from trendi.paperdoll import pd_falcon_client
from trendi.utils import imutils
from trendi import Utils,constants
from trendi.paperdoll import pd
import cv2

def get_pd_results(url):
    print('getting pd results for '+url)
    image = Utils.get_cv2_img_array(url)
    if image is None:
        print('image came back none')
    imgfilename = 'orig.jpg'
    cv2.imwrite(imgfilename,image)
    seg_res = pd_falcon_client.pd(image)
    print('segres:'+str(seg_res))
    mask = seg_res['mask']
    label_dict = seg_res['label_dict']
    pose = seg_res['pose']
    mask_np = np.array(mask, dtype=np.uint8)
    pose_np = np.array(pose, dtype=np.uint8)
    pd.convert_and_save_results(mask_np, label_dict, pose_np, imgfilename, image, url)


    maskfilename = 'testout.png'
    cv2.imwrite(maskfilename,seg_res)
    imutils.show_mask_with_labels(maskfilename,constants.fashionista_categories_augmented_zero_based,original_image = imgfilename,save_images=True)

if __name__ == "__main__":
    url = 'https://s-media-cache-ak0.pinimg.com/736x/3a/85/79/3a857905d8814faf49910f9c2b9806a8.jpg'
    get_pd_results(url)