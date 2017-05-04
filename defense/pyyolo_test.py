__author__ = 'jeremy'

import pyyolo
import cv2
import hashlib
import time

from trendi import constants

#get yolo net and keep it in mem
#datacfg = 'cfg/coco.data'
datacfg = '/data/jeremy/pyyolo/darknet/cfg/coco.data'
cfgfile = '/data/jeremy/pyyolo/darknet/cfg/tiny-yolo.cfg'
weightfile = '/data/jeremy/pyyolo/tiny-yolo.weights'
#filename = 'data/person.jpg'
thresh = 0.24
hier_thresh = 0.5
pyyolo.init(datacfg, cfgfile, weightfile)
# cam = cv2.VideoCapture(-1)
# ret_val, img = cam.read()
# print(ret_val)
# ret_val = cv2.imwrite(filename,img)
# print(ret_val)



def detect_yolo_pyyolo(self, img_arr, url='',classes=constants.hls_yolo_categories):
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
        elements = output.split()
        print('elements: '+str(elements))
        label = elements[0]
        xmin = int(elements[1])
        ymin = int(elements[2])
        xmax = int(elements[3])
        ymax = int(elements[4])
        item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':'>'+str(thresh)}
#            item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':round(float(confidence),3)}
        if elements[0] == 'person':
            cropped_image = img_arr[ymin:ymax, xmin:xmax]
            # print('crop:{} {}'.format(item["bbox"],cropped_image.shape))
            # get hydra results
            try:
                hydra_output = self.get_hydra_output(cropped_image)
                if hydra_output:
                    item['details'] = hydra_output
            except:
                print "Hydra failed " + traceback.format_exc()

        relevant_bboxes.append(item)
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
    return relevant_boxes
