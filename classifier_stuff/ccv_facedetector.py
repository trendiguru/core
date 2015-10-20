__author__ = 'jeremy'

#import Utils
import subprocess

import commands
import  cv2
import os

def ccv_facedetect(filename):
 #   Utils.get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, download=True,)
    fcommand = './ccvface '+str(filename)+' ccvface.sqlite3'
    command = './ccvface '
              #+str(filename)+' ccvface.sqlite3'
    arg1 = str(filename)
    arg2 = "ccvface.sqlite3"
#    retval = subprocess.call([fcommand],shell=True)
#    subprocess.check_call([fcommand], stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)
    #subprocess.check_output([command,arg1,arg2])
#    output = subprocess.Popen(fcommand, stdout=subprocess.PIPE).communicate()[0]
    retvals = commands.getstatusoutput(fcommand)
#    print retvals
    rects = []
    for rectstr in retvals[1:]:
        newrect = [int(s) for s in rectstr.split() if s.isdigit()]
        if len(newrect) == 4:
            rects.append(newrect)
    return rects

def check_lfw(use_visual_output=False):
    BASE_PATH = os.getcwd()
    BASE_PATH2 = os.path.join(BASE_PATH, 'male-female/female')
    print('basepath:' + BASE_PATH2)
    n_images = 0
    n_extra = 0
    n_single_detections = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH2):
        dirnames.sort()
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print('path:' + abs_path)
                faces = ccv_facedetect(abs_path)
                print('faces:'+str(faces))
                n_images = n_images + 1
                if len(faces)>1:
                    n_extra = n_extra + 1
                if len(faces)==1:
                    n_single_detections = n_single_detections + 1
                if use_visual_output:
                    show_rects(abs_path,faces)
                print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections))

    true_positives = n_single_detections + n_extra
    true_pos_rate = float(true_positives)/n_images
    false_neg_rate = float(n_images-true_positives)/n_images
    print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections))
    print('true pos:'+str(true_positives)+' false_neg:'+str(false_neg_rate))

    BASE_PATH = os.getcwd()
    BASE_PATH2 = os.path.join(BASE_PATH, 'male-female/male')
    print('basepath:' + BASE_PATH2)
    n_images = 0
    n_extra = 0
    n_single_detections = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH2):
        dirnames.sort()
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print('path:' + abs_path)
                faces = ccv_facedetect(abs_path)
                print('faces:'+str(faces))
                n_images = n_images + 1
                if len(faces)>1:
                    n_extra = n_extra + 1
                if len(faces)==1:
                    n_single_detections = n_single_detections + 1
                if use_visual_output:
                    show_rects(abs_path,faces)
                print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections))

    true_positives = n_single_detections + n_extra
    true_pos_rate = float(true_positives)/n_images
    false_neg_rate = float(n_images-true_positives)/n_images
    print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections))
    print('true pos:'+str(true_positives)+' false_neg:'+str(false_neg_rate))

def show_rects(abs_path,faces):
    img_arr = cv2.imread(abs_path)
    if len(faces):
        for rect in faces:
            print('rect:'+str(rect))
            cv2.rectangle(img_arr,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),[255,255,255],5)
            cv2.rectangle(img_arr,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),[255,255,255],5)
    cv2.imshow('candidate',img_arr)
    cv2.waitKey(10)


if __name__ == "__main__":
    check_lfw()
    filename = "../images/male1.jpg"
    faces = ccv_facedetect(filename)
    print('faces:'+str(faces))
    img_arr = cv2.imread(filename)
    if len(faces):
        for rect in faces:
            print('rect:'+str(rect))
            cv2.rectangle(img_arr,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),[255,255,255],5)
            cv2.rectangle(img_arr,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),[255,255,255],5)
    cv2.imshow('candidate',img_arr)

