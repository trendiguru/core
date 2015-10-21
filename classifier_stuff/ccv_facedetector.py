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

import Utils

def ccv_facedetect(filename):
 #   Utils.get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, download=True,)
    if not os.path.isfile(filename):
        logging.warning('file passed to ccv_facedetect doesnt exist)')
        return 1
    fcommand = 'pwd'
    retvals = commands.getstatusoutput(fcommand)
    print(str(retvals),end="\n")
    d=Utils.depth_of_subdir_of_calling_function()
    print('depth of subdir:'+str(d))
    if d == 0:
        fcommand = 'classifier_stuff/ccvface '+str(filename)+' classifier_stuff/ccvface.sqlite3'
    elif d == 1:
        fcommand = './ccvface '+str(filename)+' ./ccvface.sqlite3'
    print('command:'+fcommand)
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

def run_classifier_on_dir_of_dirs(BASE_PATH=None,use_visual_output=False,classifier=ccv_facedetect):
    if BASE_PATH is None:
        BASE_PATH = os.getcwd()
    print('basepath:' + BASE_PATH)
    n_images = 0
    n_extra = 0
    n_single_detections = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for filename in filenames:
            abs_path = "%s/%s" % (dirname, filename)
            print('path:' + abs_path)
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

        raw_input('waiting')
        dirnames.sort()
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print('path:' + abs_path)
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

    else:
        return 0,0

 #!/usr/bin/python
     # Creating an empty list that will contain the already traversed paths
def direct(path):
    donePaths = []
    for paths,dirs,files in os.walk(path):
        if paths not in donePaths:
            count = paths.count('/')
            if files:
                for ele1 in files:
                    print('---------' * (count), ele1)
                if dirs:
                    for ele2 in dirs:
                        print('---------' * (count), ele2)
                        absPath = os.path.join(paths,ele2)
          # recursively calling the direct function on each directory
                        direct(absPath)
               # adding the paths to the list that got traversed
                        donePaths.append(absPath)


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

#    direct('.')
    pos,neg = run_classifier_on_dir_of_dirs('/home/developer/python_packages/trendi_guru_modules/classifier_stuff/images/llamas')
    print('pos:{0} neg:{1}'.format(pos,neg))
    pos,neg = run_classifier_on_dir_of_dirs('/home/developer/python_packages/trendi_guru_modules/classifier_stuff/images/monkeys')
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

