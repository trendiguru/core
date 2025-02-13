#this file has to go in the rcnn folder
#swapping yolo into place instead of this, use 8082 for that and not this so roy doesnt have to change anything
"""
run this like:
gunicorn -b :8082 -w 1 -k gevent -n hls --timeout 120 trendi.defense.defense_falcon_rcnn:api
assuming the docker was started with port 8082 specified e.g.
nvidia-docker run -it -v /data:/data -p 8082:8082 --name frcnn eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'
"""
#this file has to go in the rcnn folder


import traceback
import falcon
from falcon_cors import CORS
import subprocess
import os
import defense_rcnn
import requests
import hashlib
import time
from jaweson import json, msgpack
import numpy as np
import cv2


from trendi import constants
from trendi import Utils
from trendi.utils import imutils
#from trendi.defense import pyyolo_results

# print('falcon is coming form '+str(falcon.__file__))
# base_dir = os.path.dirname(os.path.realpath(__file__))
# print('current_dir is '+str(base_dir))

print "Done with imports"



# Containers must be on the same docker network for this to work (otherwise go backt o commented IP address
HYDRA_CLASSIFIER_ADDRESS = "http://hls_hydra:8081/hydra" # constants.HYDRA_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8081/hydra"
FRCNN_CLASSIFIER_ADDRESS = constants.FRCNN_CLASSIFIER_ADDRESS # "http://13.82.136.127:8082/hls"
# what is the frcnn referring to - maybe its the thing at the end of file
# namely, api.add_route('/frcnn/', HydraResource())

class HLS:
    def __init__(self):
        print "Loaded Resource"


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
        print('params into on_get: url {} x1 {} x2 {} y1 {} y2 {} net {}'.format(image_url,r_x1,r_x2,r_y1,r_y2,net))
        if not image_url:
            print('get request:' + str(req) + ' is missing imageUrl param')
            raise falcon.HTTPMissingParam("imageUrl")
        else:
            try:
                response = requests.get(image_url)
                img_arr = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
                if r_x1 or r_x2 or r_y1 or r_y2:
                    img_arr = img_arr[r_y1:r_y2, r_x1:r_x2]
                    print "ROI: {},{},{},{}; img_arr.shape: {}".format(r_x1, r_x2, r_y1, r_y2, str(img_arr.shape))
                #which net to use - yolo or rcnn, default to yolo
                if not net:
                    detected = self.detect_yolo_shell(img_arr, url=image_url)
                elif net == "yolo":
                    detected = self.detect_yolo_shell(img_arr, url=image_url)
                elif net == "rcnn":
                    detected = self.detect_rcnn(img_arr, url=image_url)
                else:
                    detected = self.detect_yolo_shell(img_arr, url=image_url)
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
        #NOTE this doesnt run yolo , only frcnn - call detect_yolo instead of detect_frcnnif we need yolo here
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
            detected = self.detect_rcnn(img_arr)
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


    def detect_rcnn(self, img_arr, url=''):
        print('started defense_falcon_rcnn.detect_rcnn')
        detected = defense_rcnn.detect_frcnn(img_arr)
        for item in detected:
            cat = item["object"]
            if cat == "person":
                print('bbox:'+str(item['bbox'])+' type:'+str(type(item['bbox'])))
                x1,y1,x2,y2 = item["bbox"]
                print('x1 {} y1 {} x2 {} y2 {} type {}:'.format(x1,y1,x2,y2,type(x1)))
                print('img arr type:'+str(type(img_arr)))
                print('img arr shape:'+str((img_arr.shape)))
                cropped_image = img_arr[y1:y2, x1:x2]
                # print('crop:{} {}'.format(item["bbox"],cropped_image.shape))
                # get hydra results
                try:
                    hydra_output = self.get_hydra_output(cropped_image)
                    if hydra_output:
                        item['details'] = hydra_output
                except:
                    print "Hydra failed " + traceback.format_exc()
        self.write_log(url,detected)
        print detected
        return detected


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

api.add_route('/hls/', HLS())
