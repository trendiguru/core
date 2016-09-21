__author__ = 'jeremy'



#example for using this
#from . import mnc_voc_pixlevel_segmenter
import time
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from trendi.paperdoll import mnc_falcon_client
import urllib
import argparse
import time
import cv2
import numpy as np
# User-defined module
# import _init_paths
from . import mnc_init_path
import caffe

#imports done in original demo.py - these require MNC to be in pythonpath
#from mnc_config import cfg
#from transform.bbox_transform import clip_boxes
#from utils2.blob import prep_im_for_blob, im_list_to_blob
#from transform.mask_transform import gpu_mask_voting
#from utils2.vis_seg import _convert_pred_to_image, _get_voc_color_map
#import matplotlib.pyplot as plt
#import Image
#import urllib
#import time


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    if url.count('jpg') > 1:
        return None
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return new_image


def prepare_mnc_args(im, net):
    # Prepare image data blob
    blobs = {'data': None}
    processed_ims = []
    im, im_scale_factors = \
        prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TEST.SCALES[0], cfg.TRAIN.MAX_SIZE)
    processed_ims.append(im)
    blobs['data'] = im_list_to_blob(processed_ims)
    # Prepare image info blob
    im_scales = [np.array(im_scale_factors)]
    assert len(im_scales) == 1, 'Only single-image batch implemented'
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)
    # Reshape network inputs and do forward
    net.blobs['data'].reshape(*blobs['data'].shape)
    net.blobs['im_info'].reshape(*blobs['im_info'].shape)
    forward_kwargs = {
        'data': blobs['data'].astype(np.float32, copy=False),
        'im_info': blobs['im_info'].astype(np.float32, copy=False)
    }
    return forward_kwargs, im_scales


def im_detect(im, net):
    forward_kwargs, im_scales = prepare_mnc_args(im, net)
    blobs_out = net.forward(**forward_kwargs)
    # output we need to collect:
    # 1. output from phase1'
    rois_phase1 = net.blobs['rois'].data.copy()
    masks_phase1 = net.blobs['mask_proposal'].data[...]
    scores_phase1 = net.blobs['seg_cls_prob'].data[...]
    # 2. output from phase2
    rois_phase2 = net.blobs['rois_ext'].data[...]
    masks_phase2 = net.blobs['mask_proposal_ext'].data[...]
    scores_phase2 = net.blobs['seg_cls_prob_ext'].data[...]
    # Boxes are in resized space, we un-scale them back
    rois_phase1 = rois_phase1[:, 1:5] / im_scales[0]
    rois_phase2 = rois_phase2[:, 1:5] / im_scales[0]
    rois_phase1, _ = clip_boxes(rois_phase1, im.shape)
    rois_phase2, _ = clip_boxes(rois_phase2, im.shape)
    # concatenate two stages to get final network output
    masks = np.concatenate((masks_phase1, masks_phase2), axis=0)
    boxes = np.concatenate((rois_phase1, rois_phase2), axis=0)
    scores = np.concatenate((scores_phase1, scores_phase2), axis=0)
    return boxes, masks, scores


def get_vis_dict(result_box, result_mask, img_name, cls_names, vis_thresh=0.5):
    box_for_img = []
    mask_for_img = []
    cls_for_img = []
    for cls_ind, cls_name in enumerate(cls_names):
        det_for_img = result_box[cls_ind]
        seg_for_img = result_mask[cls_ind]
        keep_inds = np.where(det_for_img[:, -1] >= vis_thresh)[0]
        for keep in keep_inds:
            box_for_img.append(det_for_img[keep])
            mask_for_img.append(seg_for_img[keep][0])
            cls_for_img.append(cls_ind + 1)
    res_dict = {'image_name': img_name,
                'cls_name': cls_for_img,
                'boxes': box_for_img,
                'masks': mask_for_img}
    return res_dict

def mnc_pixlevel_detect(url_or_np_array):
    demo_dir = './'
    start = time.time()
    if isinstance(url_or_np_array,basestring):
        im = url_to_image(url_or_np_array)
        im_name = url_or_np_array.split('/')[-1]
    else:
        im = url_or_np_array
        im_name=int(time.time())+'.jpg'
    #resize to max dim of max_dim
    max_dim = 400
    h,w = im.shape[0:2]
    compress_factor = float(max(h,w))/max_dim
    new_h = int(float(h)/compress_factor)
    new_w = int(float(w)/compress_factor)
    im = cv2.resize(im,(new_w,new_h))
    actual_new_h,actual_new_w = im.shape[0:2]
    print('old w,h {}x{}, planned {}x{}, actual {}x{}'.format(w,h,new_w,new_h,actual_new_w,actual_new_h))

    gt_image = os.path.join(demo_dir, im_name)
    print gt_image
    cv2.imwrite(gt_image,im)

    boxes, masks, seg_scores = im_detect(im, net)
    end = time.time()
    print 'im_detect time %f' % (end-start)
    start = time.time()
    result_mask, result_box = gpu_mask_voting(masks, boxes, seg_scores, len(CLASSES) + 1,
                                               100, im.shape[1], im.shape[0])
    end = time.time()

    return result_mask,result_box,im,im_name


def get_mnc_output_using_falcon(url):
    demo_dir = './'

    result_dict = mnc_falcon_client.mnc(url, cat_to_look_for='person')
    print('dict from falcon dict:'+str(result_dict))
    if not result_dict['success']:
        print('did not get nfc mnc result succesfully')
        return
    mnc_output = result_dict['mnc_output']
    print('mnc output:'+str(mnc_output))
    result_mask = mnc_output[0]
    result_box = mnc_output[1]
    superimposed_im = mnc_output[2]
    im_name = mnc_output[3]
    orig_im = mnc_output[4]
    cv2.imshow('superimpose',superimposed_im)
    return mnc_output

def extras():
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