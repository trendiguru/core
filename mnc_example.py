__author__ = 'jeremy'



#example for using this
from . import mnc_voc_pixlevel_segmenter
import time
import os
import numpy as np
import Image
import cv2
import matplotlib.pyplot as plt

def get_mnc_output_using_nfc(url):
    demo_dir = './'

    result_dict = mnc(url, cat_to_look_for='person')
    print('dict from falcon dict:'+str(result_dict))
    if not result_dict['success']:
        print('did not get nfc mnc result succesfully')
        return
    mnc_output = result_dict['mnc_output']
    print('mnc output:'+str(mnc_output))
    result_mask = mnc_output[0]
    result_box = mnc_output[1]
    im = mnc_output[2]
    im_name = mnc_output[3]

    start = time.time()
    pred_dict = mnc_voc_pixlevel_segmenter.get_vis_dict(result_box, result_mask, 'data/demo/' + im_name, CLASSES)
    end = time.time()
    print 'gpu vis dicttime %f' % (end-start)

    start = time.time()
    img_width = im.shape[1]
    img_height = im.shape[0]

    inst_img, cls_img = mnc_voc_pixlevel_segmenter._convert_pred_to_image(img_width, img_height, pred_dict)
    color_map = mnc_voc_pixlevel_segmenter._get_voc_color_map()
    target_cls_file = os.path.join(demo_dir, 'cls_' + im_name)
    cls_out_img = np.zeros((img_height, img_width, 3))
    for i in xrange(img_height):
        for j in xrange(img_width):
           cls_out_img[i][j] = color_map[cls_img[i][j]][::-1]
    cv2.imwrite(target_cls_file, cls_out_img)

    end = time.time()
    print 'convert pred to image  %f' % (end-start)

    start = time.time()
    background = Image.open(gt_image)
    mask = Image.open(target_cls_file)
    background = background.convert('RGBA')
    mask = mask.convert('RGBA')

    end = time.time()
    print 'superimpose 0 time %f' % (end-start)
    start = time.time()

    superimpose_image = Image.blend(background, mask, 0.8)
    superimpose_name = os.path.join(demo_dir, 'final_' + im_name)
    superimpose_image.save(superimpose_name, 'JPEG')
    im = cv2.imread(superimpose_name)

    end = time.time()
    print 'superimpose 1 time %f' % (end-start)
    start = time.time()

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    end = time.time()
    print 'superimpose 1.5 time %f' % (end-start)
    start = time.time()

    classes = pred_dict['cls_name']

    end = time.time()
    print 'pred_dict time %f' % (end-start)
    start = time.time()

    for i in xrange(len(classes)):
        score = pred_dict['boxes'][i][-1]
        bbox = pred_dict['boxes'][i][:4]
        cls_ind = classes[i] - 1
        ax.text(bbox[0], bbox[1] - 8,
           '{:s} {:.4f}'.format(CLASSES[cls_ind], score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(demo_dir, im_name[:-4]+'.png'))
    os.remove(superimpose_name)
    os.remove(target_cls_file)
    end = time.time()
    print 'text and save time %f' % (end-start)
#    return fig  #watch out this is returning an Image object not our usual cv2 np array


    return mnc_output #