from detector import Darknet_ObjectDetector as ObjectDetector
from detector import DetBBox

import requests
from PIL import Image
from PIL import ImageFilter
from StringIO import StringIO #

ObjectDetector.set_device(1)
det = ObjectDetector('../cfg/yolo.cfg','../yolo.weights')

def _get_image(url):
    return Image.open(StringIO(requests.get(url).content))

#we should watch out as yolo expects an array from Image.open (RGB) and not cv2.open (BGR)
def get_yolo_results(url_or_image_array):
    voc_names = ["aeroplane", "bicycle", "bird", "boat", "bottle","bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbi$
    is isinstance(url_or_image_array,basestring):
        img_arr = _get_image(url_or_image_array)
    rst, run_time = det.detect_object(img_arr)
    print 'got {} objects in {} seconds'.format(len(rst), run_time)    for bbox in rst:
        print '{} {} {} {} {} {}'.format(voc_names[bbox.cls], bbox.top, bbox.left, bbox.bottom, bbox.right, bbox.confidence)
    return rst


if __name__ == '__main__':
    from PIL import Image
    ObjectDetector.set_device(1)
    voc_names = ["aeroplane", "bicycle", "bird", "boat", "bottle","bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbi$
    det = ObjectDetector('../cfg/yolo.cfg','../yolo.weights')
    url = 'http://farm9.staticflickr.com/8323/8141398311_2fd0af60f7.jpg'
    #for i in xrange(4):
    rst, run_time = det.detect_object(_get_image(url))

    print 'got {} objects in {} seconds'.format(len(rst), run_time)

    for bbox in rst:
        print '{} {} {} {} {} {}'.format(voc_names[bbox.cls], bbox.top, bbox.left, bbox.bottom, bbox.right, bbox.confidence)













#import cv2
#import subprocess

#which gpu
gpu = 0
# -i is what gpu to use
yolo_shell_cmd = '/home/jeremy/darknet/darknet -i '+str(gpu)+' yolo test cfg/yolo.cfg yolo.weights yolo_image.jpg'

def get_shirt_output(img):
    cv2.imwrite('yolo_image.jpg')
    subprocess.call(yolo_shell_cmd,shell=True)
