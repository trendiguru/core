__author__ = 'jeremy'

import pyyolo
import cv2
import hashlib
import time
import cv2

from trendi import constants

#get yolo net and keep it in mem
#datacfg = 'cfg/coco.data'
#datacfg = '/data/jeremy/pyyolo/darknet/cfg/coco.data'
datacfg = '/data/jeremy/darknet_orig/cfg/hls.data'
cfgfile = '/data/jeremy/darknet_orig/cfg/yolo-voc_608.cfg'
#cfgfile = '/data/jeremy/pyyolo/darknet/cfg/tiny-yolo.cfg'
#weightfile = '/data/jeremy/pyyolo/tiny-yolo.weights'
weightfile = '/data/jeremy/darknet_orig/bb_hls1/yolo-voc_544_95000.weights'
thresh = 0.24
hier_thresh = 0.5
pyyolo.init(datacfg, cfgfile, weightfile)


def detect_yolo_pyyolo(img_arr, url='',classes=constants.hls_yolo_categories):
    # from file
    print('----- test original C using a file')
    hash = hashlib.sha1()
    hash.update(str(time.time()))
    img_filename = hash.hexdigest()[:10]+'pyyolo.jpg'
  #  img_filename = 'incoming.jpg'
    cv2.imwrite(img_filename,img_arr)

    outputs = pyyolo.test(img_filename, thresh, hier_thresh)
    relevant_bboxes = []
    for output in outputs:
        print(output)
        label = output['class']
        xmin = output['left']
        ymin = output['top']
        xmax = output['right']
        ymax = output['bottom']
        item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':'>'+str(thresh)}
#            item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':round(float(confidence),3)}

        relevant_bboxes.append(item)

  #not sure wht the diff is between second method and first

    # camera
    # print('----- test python API using a file')
    # i = 1
    # while i < 2:
    #     # ret_val, img = cam.read()
    #     img = cv2.imread(filename)
    #     img = img.transpose(2,0,1)
    #     c, h, w = img.shape[0], img.shape[1], img.shape[2]
    #     # print w, h, c
    #     data = img.ravel()/255.0
    #     data = np.ascontiguousarray(data, dtype=np.float32)
    #     outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
    #     for output in outputs:
    #         print(output)
    #     i = i + 1
    # free model
    pyyolo.cleanup()
    return relevant_bboxes

if __name__=="__main__":
    img_arr = cv2.imread('/data/jeremy/image_dbs/bags_for_tags/photo_10006.jpg')
    res = detect_yolo_pyyolo(img_arr)
    print(res)