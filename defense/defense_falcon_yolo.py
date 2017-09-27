__author__ = 'jeremy'
"""
run this like:
gunicorn -b :8082 -w 1 -k gevent -n hls_yolo --timeout 120 trendi.defense.defense_falcon_yolo:api
assuming the docker was started with port 8082 specified e.g.
nvidia-docker run -it -v /data:/data -p 8082:8082 --name hls_yolo eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'
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
import sys
import codecs
import pandas as pd

import pyyolo

from trendi import constants
from trendi import Utils
from trendi.utils import imutils
import base64
import random
import string

print "Defense_falcon_yolo done with imports"

#get yolo net and keep it in mem
#datacfg = 'cfg/coco.data'
#datacfg = '/data/jeremy/pyyolo/darknet/cfg/coco.data'

#run 14 aka v2
datacfg = '/data/jeremy/darknet/cfg/hls_v2.data'
cfgfile = '/data/jeremy/darknet/cfg/yolo.2.0.cfg' #higher res for smaller objects, increase subdivisions *4 to preserve memoery
weightfile = '/data/jeremy/darknet/v2/yolo_120000_run14.weights'

#run 13
datacfg = '/data/jeremy/darknet_orig/cfg/hls_v2.dat'
cfgfile = '/data/jeremy/darknet_orig/cfg/yolo-voc.2.0.cfg' #higher res for smaller objects, increase subdivisions *4 to preserve memoery
weightfile = '/data/jeremy/darknet_orig/v2/yolo-voc_100000_run13.weights'



#'v1'
datacfg = '/data/jeremy/darknet_orig/cfg/hls.data'
#cfgfile = '/data/jeremy/darknet_orig/cfg/yolo-voc_544.cfg'
cfgfile = '/data/jeremy/darknet_orig/cfg/yolo-voc_1088.cfg' #higher res for smaller objects, increase subdivisions *4 to preserve memoery
#cfgfile = '/data/jeremy/pyyolo/darknet/cfg/tiny-yolo.cfg'
#weightfile = '/data/jeremy/pyyolo/tiny-yolo.weights'
weightfile = '/data/jeremy/darknet_orig/bb_hls1/yolo-voc_544_95000.weights'

thresh = 0.1
hier_thresh = 0.5

pyyolo.init(datacfg, cfgfile, weightfile)


# Containers must be on the same docker network for this to work (otherwise go backt o commented IP address
YOLO_HLS_CLASSIFIER_ADDRESS = constants.YOLO_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8082/hls"
HYDRA_CLASSIFIER_ADDRESS = "http://hls_hydra:8081/hydra" # constants.HYDRA_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8081/hydra"

class HLS_YOLO:
    def __init__(self):
        print "Loaded Resource for HLS YOLO"


    def on_get(self, req, resp): #/
        """Handles GET requests"""
        serializer = json
        resp.content_type = "application/json"
        print('\nStarting HLS_YOLO (got a get request)')
        image_url = req.get_param("imageUrl")
        image = req.get_param("image")
        file = req.get_param("file")
        r_x1 = req.get_param_as_int("x1")
        r_x2 = req.get_param_as_int("x2")
        r_y1 = req.get_param_as_int("y1")
        r_y2 = req.get_param_as_int("y2")
        net = req.get_param("net")
        loc_thresh = req.get_param("threshold")
        loc_hier_thresh = req.get_param("hier_threshold")
#        for k,v in req.get_param.iteritems():
#            print('key {} value {}'.format(k,v))
        print('params into hls yolo on_get: url {} file {} x1 {} x2 {} y1 {} y2 {} net {} thresh {} hierthresh {}'.format(image_url,file,r_x1,r_x2,r_y1,r_y2,net,loc_thresh,loc_hier_thresh))
        if loc_thresh is not None:
            global thresh
            thresh = float(loc_thresh)
        if loc_hier_thresh is not None:
            global hier_thresh
            hier_thresh = float(loc_hier_thresh)
        elif image_url:
            try:
                response = requests.get(image_url)
                img_arr = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
                if img_arr == None:
                    print('got none for image array')
                    resp.data = serializer.dumps({"data": 'bad image at '+image_url})
                    resp.status = falcon.HTTP_200
                    return
            except:
                raise falcon.HTTPBadRequest("Something went wrong in get section 1:(", traceback.format_exc())
        elif image:
            print('getting img_arr directly')
            img_arr = pd.read_json(image,orient='values')
            print('img size {}'.format(img_arr.shape))
        elif file:
            print('getting file {}'.format(file))
            if not os.path.exists(file):
                raise falcon.HTTPBadRequest("could not get file "+str(file), traceback.format_exc())
            img_arr = cv2.imread(file)
            print('img size {}'.format(img_arr.shape))
        else:
            print('get request to hls yolo:' + str(req) + ' is missing both imageUrl and image param')
            raise falcon.HTTPMissingParam("imageUrl,image")
        try:
            if r_x1 or r_x2 or r_y1 or r_y2:
                img_arr = img_arr[r_y1:r_y2, r_x1:r_x2]
                print "ROI: {},{},{},{}; img_arr.shape: {}".format(r_x1, r_x2, r_y1, r_y2, str(img_arr.shape))
        except:
            raise falcon.HTTPBadRequest("Something went wrong in get section 2:(", traceback.format_exc())
        try:
            #which net to use - pyyolo or shell yolo , default to pyyolo
            if not net:
                detected = self.detect_yolo_pyyolo(img_arr, url=image_url)
            elif net == "shell":
                detected = self.detect_yolo_shell(img_arr, url=image_url)
            elif net == "pyyolo":
                detected = self.detect_yolo_pyyolo(img_arr, url=image_url)
            else:
                detected = self.detect_yolo_shell(img_arr, url=image_url)
            if (r_x1, r_y1) != (0, 0):
                for obj in detected:
                    try:
                        x1, y1, x2, y2 = obj["bbox"]
                        obj["bbox"] = x1 + r_x1, y1 + r_y1, x2 + r_x1, y2 + r_y1
                    except (KeyError, TypeError):
                        print "No valid 'bbox' in detected"
        except:
            raise falcon.HTTPBadRequest("Something went wrong in get section 3:(", traceback.format_exc())
        try:
            resp.data = serializer.dumps({"data": detected})
            resp.status = falcon.HTTP_200
        except:
            raise falcon.HTTPBadRequest("Something went wrong in get section 4:(", traceback.format_exc())

    def on_post(self, req, res):
        #untested
        print('\nStarting combine_gunicorn (got a post request)')
        start_time=time.time()
        tmpfile = '/data/jeremy/image_dbs/variant/viettel_demo/'
        N=10
        randstring = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
        tmpfile = os.path.join(tmpfile,randstring+'.jpg')

        sent_timestamp=0
 #      serializer = msgpack
 #      res.content_type = "application/x-msgpack"
        try:
            json_data = json.loads(req.stream.read().decode('utf8'))
            if 'sent_timestamp' in json_data:
                sent_timestamp = float(json_data['sent_timestamp'])
                print('sent timestamp {}'.format(sent_timestamp))
            else:
                sent_timestamp=0 #
            xfer_time = time.time()-sent_timestamp
            print('xfer time:{}'.format(xfer_time))
            base64encoded_image = json_data['image_data']
            data = base64.b64decode(base64encoded_image)
#            print('data type {}'.format(type(data)))
            with open(tmpfile, "wb") as fh:
                fh.write(data)
                print('wrote file to {}, elapsed time for xfer {}'.format(tmpfile,time.time()-start_time))
                decode_time = time.time()-start_time
            try:
                print('db2')
#                imgpath = '/data/jeremy/tensorflow/tmp.jpg'
#                tmpfile = cv2.imwrite(imgpath,img_arr)
                img_arr=cv2.imread(tmpfile)
                detected = self.detect_yolo_pyyolo(img_arr)

    #            detected = tf_detect.analyze_image(tmpfile,thresh=0.2)

            except:
                raise falcon.HTTPBadRequest("Something went wrong in get section 3:(", traceback.format_exc())
            try:
                print('db4')
                res.data = json.dumps(detected)
                res.status = falcon.HTTP_200
            except:
                raise falcon.HTTPBadRequest("Something went wrong in get section 4:(", traceback.format_exc())
            try:
                self.write_log('id',detected)
            except:
                raise falcon.HTTPBadRequest("Something went wrong in get section 5 (wrte log):(", traceback.format_exc())

#             stream = req.stream.read()
#             print('stream {}'.format(stream))
#             data = serializer.loads(stream)
#             print('data:{}'.format(data))
#             img_arr = data.get("image")
#             print('img arr shape {}'.format(img_arr.shape))
# #            detected = self.detect_yolo_pyyolo(img_arr)
#             cv2.imwrite(tmpfile,img_arr)
#             detected = self.tracker.next_frame(tmpfile)
#             resp.data = serializer.dumps({"data": detected})
#             resp.status = falcon.HTTP_200

        except:
            raise falcon.HTTPBadRequest("Something went wrong in post :(", traceback.format_exc())

        res.status = falcon.HTTP_203
#        res.body = json.dumps({'status': 1, 'message': 'success','data':json.dumps(detected)})
        res.body = json.dumps(detected)







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
            return []
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
        print('started pyyolo detect, thresh='+str(thresh)+' hier '+str(hier_thresh))
        save_path = '/data/jeremy/pyyolo/results/'
        if img_arr is None:
            print('got None img array!!')
            None
        if len(img_arr.shape) == 2: #got 1-chan(gray) image
            print('got gray img')
            img_arr_bgr=np.zeros([img_arr.shape[0],img_arr.shape[1],3])
            img_arr_bgr=img_arr
            print('sizes: {} {}'.format(img_arr_bgr,img_arr))
            img_arr = img_arr_bgr
        print('img arr size {}'.format(img_arr.shape))
        max_image_size=1200 #1280 fails
        resized=False
        if img_arr.shape[0]>max_image_size or img_arr.shape[1]>max_image_size:  #maybe this is causing prob at http://a.abcnews.com/images/US/150815_yt_phillypd_33x16_1600.jpg
            maxside=max(img_arr.shape)
            reduction_factor = maxside/1000.0 #force maxside to 1000
            original_size=img_arr.shape
            resized=True
            img_arr = cv2.resize(img_arr,(int(img_arr.shape[1]/reduction_factor),int(img_arr.shape[0]/reduction_factor)))
            print('reduced size to {}'.format(img_arr.shape))
            #generate randonm filename
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        if save_results:
            timestr = time.strftime("%Y%m%d.%H%M%S")
            img_filename = timestr+hash.hexdigest()[:5]+'.jpg'
            Utils.ensure_dir(save_path)
            img_path = os.path.join(save_path,img_filename)
            print('detect_yolo_pyyolo saving file '+str(img_path))
            try:
                cv2.imwrite(img_path,img_arr)
            except:
                print('some trouble saving image,'+str(sys.exc_info()[0]))
        relevant_items = []
        print('getting results from get_pyyolo_results')
        yolo_results = self.get_pyyolo_results(img_arr)
        print('got results from get_pyyolo_results')

        for item in yolo_results:
            print(item)
            if resized:
                img_arr = cv2.resize(img_arr,(original_size[1],original_size[0]))
                item['bbox'][0]=int(item['bbox'][0]*reduction_factor)
                item['bbox'][1]=int(item['bbox'][1]*reduction_factor)
                item['bbox'][2]=int(item['bbox'][2]*reduction_factor)
                item['bbox'][3]=int(item['bbox'][3]*reduction_factor)
                print('fixed size back to original {}'.format(item))

            xmin=item['bbox'][0]
            ymin=item['bbox'][1]
            xmax=item['bbox'][2]
            ymax=item['bbox'][3]
            assert xmin<xmax,'xmin not < xmax!!!'
            assert ymin<ymax,'xmin not < xmax!!!'
##### TAKING OUT RELEVANT ITEMS ON ROYS SUGGESTION
            use_hydra=False
            if use_hydra:
                if item['object'] == 'person':
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
            if save_results:
                imutils.bb_with_text(img_arr,[xmin,ymin,(xmax-xmin),(ymax-ymin)],item['object'])
        if save_results:
            marked_imgname = img_path.replace('.jpg','_bb_yolos.jpg')
            print('pyyolo bbs being writtten to '+str(marked_imgname))
            try:
                pass  #no real need to save bb file as no extra info there
              #  r=cv2.imwrite(marked_imgname,img_arr)
              #skip saving bb image to conserve space
              #  print('write result '+str(r))
            except:
                print('some trouble saving bb image,'+str(sys.exc_info()[0]))
            txtname=img_path.replace('.jpg','.txt')
            self.write_log(url,relevant_items,filename=txtname)

        print('detect yolo returning:'+str(relevant_items))
        return relevant_items


    def get_pyyolo_results(self,img_arr, url='',classes=constants.hls_yolo_categories,method='file'):
        # from file
        relevant_bboxes = []
        if method == 'file':
            print('----- test original C using a file')
            hash = hashlib.sha1()
            hash.update(str(time.time()))
            img_filename = hash.hexdigest()[:10]+'pyyolo.jpg'
          #  img_filename = 'incoming.jpg'
            cv2.imwrite(img_filename,img_arr)
            outputs = pyyolo.test(img_filename, thresh, hier_thresh)

  #not sure what the diff is between this second method (pyyolo.detect) and first (pyyolo.test)
                #except it uses array instead of file
    # camera
    # print('----- test python API using a file')
        else:
            i = 1  #wtf is this count for
            while i < 2:
                # ret_val, img = cam.read()
#                img = cv2.imread(filename)
                img = img_arr.transpose(2,0,1)
                c, h, w = img.shape[0], img.shape[1], img.shape[2]
                # print w, h, c
                data = img.ravel()/255.0
                data = np.ascontiguousarray(data, dtype=np.float32)
                print('calling pyyolo.detect')
                try:
                    outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
                except:
                    print('some trouble calling pyyolo detect,'+str(sys.exc_info()[0]))

                print('returned from  pyyolo.detect')
                for output in outputs:
                    print(output)
                i = i + 1
    # free model

        for output in outputs:
            print(output)
            label = output['class']
            if 'person' in label:
                label='person'  #convert 'person_wearing_red/blue_shirt' into just person
            xmin = output['left']
            ymin = output['top']
            xmax = output['right']
            ymax = output['bottom']
            conf = output['conf']
            item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':round(conf,4)}
            h,w=img_arr.shape[0:2]
            frac = 5
            cropped_arr = img_arr[h/frac:h-(h/frac),w/frac:w-(w/frac)]
            dominant_color = imutils.dominant_colors(cropped_arr)
            print('dominant color:'+str(dominant_color))
            if dominant_color is not None:
                item['details']={'color':dominant_color}
    #            item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':round(float(confidence),3)}
            relevant_bboxes.append(item)

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


    def write_log(self, url, output,filename='/data/jeremy/pyyolo/results/bbs.txt'):
#        logfile = '/data/jeremy/caffenets/hydra/production/hydra/logged_hls_output.txt'
        print('logging output to '+filename)
        out = {'output':output,'url':url}
        with open(filename, 'w+') as fp:
            print('writing :'+str(out))
           # output.append = {'url':url}
            json.dump(out, fp, indent=4)
            fp.close()
    #            fp.write()



cors = CORS(allow_all_headers=True, allow_all_origins=True, allow_all_methods=True)
api = falcon.API(middleware=[cors.middleware])

api.add_route('/hls/', HLS_YOLO())
# if __name__=="__main__":
#     img_arr = cv2.imread('/data/jeremy/image_dbs/bags_for_tags/photo_10006.jpg')
#     res = detect_yolo_pyyolo(img_arr)
#     print(res)
