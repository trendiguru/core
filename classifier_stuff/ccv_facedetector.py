from __future__ import print_function

__author__ = 'jeremy'

#import Utils
import subprocess

import commands
import  cv2
import os
import sys
import os.path
import logging

def ccv_facedetect(filename):
 #   Utils.get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, download=True,)
    if not os.path.isfile(filename):
        logging.warning('file passed to ccv_facedetect doesnt exist)')
        return 1
    fcommand = 'pwd'
    retvals = commands.getstatusoutput(fcommand)
    print(str(retvals),end="\n")
    d=depth_of_subdir_in_trendi_guru_modules_of_calling_function()
    print('depth of subdir:'+str(d))

    fcommand = 'classifier_stuff/ccvface '+str(filename)+' classifier_stuff/ccvface.sqlite3'
    retvals = commands.getstatusoutput(fcommand)
    print(str(retvals),end="\n")

#    command = './ccvface '
              #+str(filename)+' ccvface.sqlite3'
    arg1 = str(filename)
    arg2 = "ccvface.sqlite3"
#    retval = subprocess.call([fcommand],shell=True)
#    subprocess.check_call([fcommand], stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)
    #subprocess.check_output([command,arg1,arg2])
#    output = subprocess.Popen(fcommand, stdout=subprocess.PIPE).communicate()[0]
    retvals = commands.getstatusoutput(fcommand)
    print(str(retvals),end="\n")
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

                faces = ccv_facedetect(abs_path)
                print('path:' + abs_path+' faces:'+str(faces), end="\r")
                n_images = n_images + 1
                if len(faces)>1:
                    n_extra = n_extra + 1
                if len(faces)==1:
                    n_single_detections = n_single_detections + 1
                if use_visual_output:
                    show_rects(abs_path,faces)
                print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections), end="\r")

    true_positives = n_single_detections + n_extra
    true_pos_rate = float(true_positives)/n_images
    false_neg_rate = float(n_images-true_positives)/n_images
    print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections))
    print('true pos:'+str(true_pos_rate)+' false_neg:'+str(false_neg_rate))

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

                faces = ccv_facedetect(abs_path)
                print('path:' + abs_path+' faces:'+str(faces), end="\r")
                n_images = n_images + 1
                if len(faces)>1:
                    n_extra = n_extra + 1
                if len(faces)==1:
                    n_single_detections = n_single_detections + 1
                if use_visual_output:
                    show_rects(abs_path,faces)
                print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections), end="\n")

    true_positives = n_single_detections + n_extra
    true_pos_rate = float(true_positives)/n_images
    false_neg_rate = float(n_images-true_positives)/n_images
    print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections))
    print('true pos:'+str(true_pos_rate)+' false_neg:'+str(false_neg_rate))

def run_classifier_on_dir_recursively(BASE_PATH=None,use_visual_output=False,classifier=ccv_facedetect):
    if BASE_PATH is None:
        BASE_PATH = os.getcwd()
    print('basepath:' + BASE_PATH)
    n_images = 0
    n_extra = 0
    n_single_detections = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        dirnames.sort()
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                faces = classifier(abs_path)
                print('path:' + abs_path+' faces:'+str(faces), end="\r")
                n_images = n_images + 1
                if len(faces)>1:
                    n_extra = n_extra + 1
                if len(faces)==1:
                    n_single_detections = n_single_detections + 1
                if use_visual_output:
                    show_rects(abs_path,faces)
                print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections), end="\r")

    if n_images:
        positives = n_single_detections + n_extra
        pos_rate = float(true_positives)/n_images
        neg_rate = float(n_images-true_positives)/n_images
        print('n_images:'+str(n_images)+' n_extra:'+str(n_extra)+' n_detections:'+str(n_single_detections))
        print('true pos:'+str(pos_rate)+' false_neg:'+str(neg_rate))
        return pos_rate,neg_rate

def show_rects(abs_path,faces):
    img_arr = cv2.imread(abs_path)
    if len(faces):
        for rect in faces:
            print('rect:'+str(rect))
            cv2.rectangle(img_arr,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),[255,255,255],5)
            cv2.rectangle(img_arr,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),[255,255,255],5)
    cv2.imshow('candidate',img_arr)
    cv2.waitKey(10)


def depth_of_subdir_in_trendi_guru_modules_of_calling_function():
    path = os.getcwd()
    print('path:'+str(path))
    p2 = path.split('trendi_guru_modules/')
    print('path split on trendigurumodules:'+str(p2))
    if len(p2) < 2:
        print('not in trendi_guru_modules')
    secondhalf = p2[1]
    print('secondhalf:'+str(secondhalf))
    cur = secondhalf.split('/')
    print('cur:'+str(cur))
    if len(cur) > 1:
        in_subdir_of_trendi_guru_modules = True
    return len(cur)-1

if __name__ == "__main__":

    a=depth_of_subdir_in_trendi_guru_modules_of_calling_function()
    print('depth = '+str(a))
    raw_input('k')

    pos,neg = run_classifier_on_dir_recursively('images/male_faces')
    print('pos:{0} neg:{1}'.format(pos,neg))

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

