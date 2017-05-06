__author__ = 'jeremy'
"""
run this like:
gunicorn -b :8084 -w 1 -k gevent -n hls_yolo --timeout 120 trendi.defense.defense_falcon_yolo:api
assuming the docker was started with port 8084 specified e.g.
nvidia-docker run -it -v /data:/data -p 8084:8084 --name hls_yolo eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'
"""

import falcon
from falcon_cors import CORS

import traceback
import subprocess
import os
import requests
import hashlib
import time
from jaweson import json, msgpack
import numpy as np
import cv2

import pyyolo

from trendi import constants
from trendi import Utils
from trendi.utils import imutils

print "Defense_falcon_yolo done with imports"

#get yolo net and keep it in mem
#datacfg = 'cfg/coco.data'
#datacfg = '/data/jeremy/pyyolo/darknet/cfg/coco.data'
datacfg = '/data/jeremy/darknet_orig/cfg/hls.data'
cfgfile = '/data/jeremy/darknet_orig/cfg/yolo-voc_544.cfg'
#cfgfile = '/data/jeremy/pyyolo/darknet/cfg/tiny-yolo.cfg'
#weightfile = '/data/jeremy/pyyolo/tiny-yolo.weights'
weightfile = '/data/jeremy/darknet_orig/bb_hls1/yolo-voc_544_95000.weights'
thresh = 0.24
hier_thresh = 0.5
pyyolo.init(datacfg, cfgfile, weightfile)


# Containers must be on the same docker network for this to work (otherwise go backt o commented IP address
YOLO_HLS_CLASSIFIER_ADDRESS = constants.YOLO_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8083/hls"
HYDRA_CLASSIFIER_ADDRESS = "http://hls_hydra:8081/hydra" # constants.HYDRA_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8081/hydra"

class HLS_YOLO:
    def __init__(self):
        print "Loaded Resource for HLS YOLO"


    def on_get(self, req, resp): #
        """Handles GET requests"""
        serializer = json
        resp.content_type = "application/json"

        image_url = req.get_param("imageUrl")
        r_x1 = req.get_param_as_int("x1")
        r_x2 = req.get_param_as_int("x2")
        r_y1 = req.get_param_as_int("y1")
        r_y2 = req.get_param_as_int("y2")
        net = req.get_param("net")
        print('params into hls yolo on_get: url {} x1 {} x2 {} y1 {} y2 {} net {}'.format(image_url,r_x1,r_x2,r_y1,r_y2,net))
        if not image_url:
            print('get request to hls yolo:' + str(req) + ' is missing imageUrl param')
            raise falcon.HTTPMissingParam("imageUrl")
        else:
            try:
                response = requests.get(image_url)
                img_arr = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
                if r_x1 or r_x2 or r_y1 or r_y2:
                    img_arr = img_arr[r_y1:r_y2, r_x1:r_x2]
                    print "ROI: {},{},{},{}; img_arr.shape: {}".format(r_x1, r_x2, r_y1, r_y2, str(img_arr.shape))
                #which net to use - pyyolo or shell yolo , default to pyyolo
                if not net:
                    detected = self.detect_yolo_pyyolo(img_arr, url=image_url)
                elif net == "shell":
                    detected = self.detect_yolo_shell(img_arr, url=image_url)
                elif net == "pyyolo":
                    detected = self.detect_yolo_pyyolo(img_arr, url=image_url)
                else:
                    detected = self.detect_yolo_pyyolo(img_arr, url=image_url)
                if (r_x1, r_y1) != (0, 0):
                    for obj in detected:
                        try:
                            x1, y1, x2, y2 = obj["bbox"]
                            obj["bbox"] = x1 + r_x1, y1 + r_y1, x2 + r_x1, y2 + r_y1
                        except (KeyError, TypeError):
                            print "No valid 'bbox' in detected"


                resp.data = serializer.dumps({"data": detected})
                resp.status = falcon.HTTP_200
            except:
                raise falcon.HTTPBadRequest("Something went wrong in get:(", traceback.format_exc())


    def on_post(self, req, resp):
        #untested
        serializer = msgpack
        resp.content_type = "application/x-msgpack"
        try:
            data = serializer.loads(req.stream.read())
            img_arr = data.get("image")
            roi = data.get("roi")
            if roi:
                r_x1, r_y1, r_x2, r_y2 = roi
                img_arr = img_arr[r_y1:r_y2, r_x1:r_x2]
                print "ROI: {},{},{},{}; img_arr.shape: {}".format(r_x1, r_x2, r_y1, r_y2, str(img_arr.shape))
            detected = self.detect_yolo_pyyolo(img_arr)
            if roi and (r_x1, r_y1) != (0, 0):
                for obj in detected:
                    x1, y1, x2, y2 = obj["bbox"]
                    obj["bbox"] = x1 + r_x1, y1 + r_y1, x2 + r_x1, y2 + r_y1
            resp.data = serializer.dumps({"data": detected})
            resp.status = falcon.HTTP_200
        except:
            raise falcon.HTTPBadRequest("Something went wrong in post :(", traceback.format_exc())


    def detect_yolo_shell(self, img_arr, url='',classes=constants.hls_yolo_categories,save_results=True):
        #RETURN dict like: ({'object':class_name,'bbox':bbox,'confidence':round(float(score),3)})
 #  relevant_bboxes.append({'object':class_name,'bbox':bbox,'confidence':round(float(score),3)})
        print('started defense_falcon_rcnn.detect_yolo')
        hash = hashlib.sha1()
        hash.update(str(time.time()))
      #  img_filename = 'incoming.jpg'
        yolo_path = '/data/jeremy/darknet_python/darknet'
#        cfg_path = '/data/jeremy/darknet_python/cfg/yolo-voc_544.cfg'
        cfg_path = '/data/jeremy/darknet_python/cfg/yolo-voc_608.cfg'
#        weights_path = '/data/jeremy/darknet_python/yolo-voc_544_95000.weights'
        weights_path = '/data/jeremy/darknet_python/yolo-voc_608_46000.weights'
        save_path = './results/'
        detections_path = 'detections.txt'  #these are getting stored in local dir it seems
        img_filename = hash.hexdigest()[:10]+'.jpg'
        Utils.ensure_dir(save_path)
        img_path = os.path.join(save_path,img_filename)
        print('save file '+str(img_path))
        cv2.imwrite(img_path,img_arr)
        try:
            os.remove(detections_path)  #this is a temp file to hold current detection - erase then write
        except:
            print('file {} doesnt exist'.format(detections_path))
        cmd = yolo_path+' detect '+cfg_path+' '+weights_path+' '+img_path
        subprocess.call(cmd, shell=True)  #blocking call
        time.sleep(0.1) #wait for file to get written
        relevant_bboxes = []
        if not os.path.exists(detections_path):
            return {'object':None}
        with open(detections_path,'r') as fp:
            lines = fp.readlines()
            fp.close()
        saved_detections = img_path.replace('.jpg','.txt')
        with open(saved_detections,'w') as fp2: #copy into another file w unique name so we can delete original
            fp2.writelines(lines)
            fp2.close()

        for line in lines:
            label_index,confidence,xmin,ymin,xmax,ymax = line.split()
            label_index=int(label_index)
            label=classes[label_index]
            confidence=float(confidence)
            xmin=int(xmin)
            xmax=int(xmax)
            ymin=int(ymin)
            ymax=int(ymax)
            item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':round(float(confidence),3)}
            if label == 'person':
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
            if save_results:
                imutils.bb_with_text(img_arr,[xmin,ymin,(xmax-xmin),(ymax-ymin)],label)
        if save_results:
            marked_imgname = img_path.replace('.jpg','_bbs.jpg')
            print('bbs writtten to '+str(marked_imgname))
            cv2.imwrite(marked_imgname,img_arr)
        self.write_log(url,relevant_bboxes)
        print relevant_bboxes
        return relevant_bboxes


    def detect_yolo_pyyolo(self, img_arr, url='',classes=constants.hls_yolo_categories,save_results=True):
#                item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':'>'+str(thresh)}
        print('started pyyolo detect')
        save_path = './results/'
    #generate randonm filename
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        img_filename = hash.hexdigest()[:10]+'.jpg'
        Utils.ensure_dir(save_path)
        img_path = os.path.join(save_path,img_filename)
        print('detecto_yolo_pyyolo saving file '+str(img_path))
        cv2.imwrite(img_path,img_arr)
        relevant_items = []
        yolo_results = self.get_pyyolo_results(img_arr)
        for item in yolo_results:
            print(item)
            if item['object'] == 'person':
                xmin=item['bbox'][0]
                ymin=item['bbox'][1]
                xmax=item['bbox'][2]
                ymax=item['bbox'][3]
                cropped_image = img_arr[ymin:ymax, xmin:xmax]
                # print('crop:{} {}'.format(item["bbox"],cropped_image.shape))
                # get hydra results
                try:
                    hydra_output = self.get_hydra_output(cropped_image)
                    if hydra_output:
                        item['details'] = hydra_output
                except:
                    print "Hydra call from pyyolo defense falcon failed " + traceback.format_exc()

            relevant_items.append(item)
#        pyyolo.cleanup()
            if save_results:
                imutils.bb_with_text(img_arr,[xmin,ymin,(xmax-xmin),(ymax-ymin)],item['object'])
        if save_results:
            marked_imgname = img_path.replace('.jpg','_bbs.jpg')
            print('pyyolo bbs writtten to '+str(marked_imgname))
            cv2.imwrite(marked_imgname,img_arr)



    def get_pyyolo_results(self,img_arr, url='',classes=constants.hls_yolo_categories):
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

  #not sure what the diff is between this second method (pyyolo.detect) and first (pyyolo.test)
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

#        pyyolo.cleanup()
        return relevant_bboxes


    def get_hydra_output(self, subimage):
        '''
        get hydra details on an image
        :param subimage: np array , e..g a crop of the original which fcrnn has found
        :return:
        '''
        data = json.dumps({"image": subimage})
        print('defense falcon is attempting to get response from hydra at ' + str(HYDRA_CLASSIFIER_ADDRESS))
        try:
            resp = requests.post(HYDRA_CLASSIFIER_ADDRESS, data=data)
            dict = json.loads(resp.content)
            return dict['output']
        except:
            print('couldnt get hydra output')
            return None


    def write_log(self, url, output):
        logfile = '/data/jeremy/caffenets/hydra/production/hydra/logged_hls_output.txt'
        print('logging output to '+logfile)
        out = {'output':output,'url':url}
        with open(logfile, 'a') as fp:
           # output.append = {'url':url}
            json.dumps(out, fp, indent=4)
#            fp.write()


cors = CORS(allow_all_headers=True, allow_all_origins=True, allow_all_methods=True)
api = falcon.API(middleware=[cors.middleware])

api.add_route('/hls_yolo/', HLS_YOLO())
# if __name__=="__main__":
#     img_arr = cv2.imread('/data/jeremy/image_dbs/bags_for_tags/photo_10006.jpg')
#     res = detect_yolo_pyyolo(img_arr)
#     print(res)
