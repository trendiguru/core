from __future__ import print_function

__author__ = 'jeremy'

#import Utils
import subprocess

import commands
import cv2
import os
import sys
import os.path
import logging

logging.basicConfig(filename='logging.log',level=logging.DEBUG)

#from .. import background_removal
#import Utils

def ccv_facedetect(filename):
 #   Utils.get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, download=True,)
    if not os.path.isfile(filename):
        logging.warning('file passed to ccv_facedetect doesnt exist)')
        return 1
#    fcommand = 'pwd'
#    retvals = commands.getstatusoutput(fcommand)
#    print(str(retvals),end="\n")
    d=depth_of_subdir_of_calling_function()
#    print('depth of subdir:'+str(d))
#    d = 1
    if d == 0:
        fcommand = 'classifier_stuff/ccvface '+str(filename)+' classifier_stuff/ccvface.sqlite3'
    elif d == 1:
        fcommand = './ccvface '+str(filename)+' ./ccvface.sqlite3'
#    print('command:'+fcommand)
    retvals = commands.getstatusoutput(fcommand)
    print('return from command '+str(fcommand)+':'+str(retvals),end="\n")
    rects = []
    if isinstance(retvals[1],basestring):
        strings = retvals[1].split('\n')
        logging.debug('strings:'+str(strings))
        for rectstr in strings:
            newrect = [int(s) for s in rectstr.split() if s.isdigit()]
            if len(newrect) == 4:
                rects.append(newrect)
                #logging.debug('rects found:'+str(rects))
            else:
                logging.warning('got weird string from ccv:'+str(rectstr))
        arr = np.asarray(rects,dtype='uint16')
        logging.debug('rects'+str(arr))
        return rects
    else:
        logging.debug('no answer string recd from ccv')
        return None

def check_lfw(use_visual_output=False):
    BASE_PATH = os.getcwd()
    BASE_PATH2 = os.path.join(BASE_PATH, 'many_faces')
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
#                faces = background_removal.find_face()


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

def run_classifier_recursively(path=None,use_visual_output=False,classifier=ccv_facedetect,n_images=0,n_single_detections=0,n_extra_detections=0,classifier_arg=None):
    if path is None:
        path = os.getcwd()
    print('basepath:' + path)
    raw_input('enter to continue')

    donePaths = []
    for paths,dirs,files in os.walk(path):
        if paths not in donePaths:
            count = paths.count('/')
            if files:
                for ele1 in files:
                    raw_input('enter to continue')
#                  print('---------' * (count), ele1)
                    full_name = os.path.join(path,ele1)
                    print('arg to classifier:'+str(classifier_arg))
                    img_arr = cv2.imread(full_name)
                    faces = classifier(img_arr,method=classifier_arg)
                    n_images = n_images + 1
                    print('faces:'+str(faces)+' images:'+str(n_images)+ ' file:'+str(ele1), end="\n")
                    if len(faces)>1:
                        n_extra_detections = n_extra_detections + len(faces) - 1
                    if len(faces)==1:
                        n_single_detections = n_single_detections + 1
                    if use_visual_output:
                        show_rects(full_name,faces)
                    print('n_images:'+str(n_images)+' n_extra:'+str(n_extra_detections)+' n_detections:'+str(n_single_detections)+' file:'+str(ele1), end="\n")
			print('')
                if dirs:
                    for ele2 in dirs:
                        print('---------' * (count), ele2)
                        absPath = os.path.join(paths,ele2)
          # recursively calling the direct function on each directory
                        n_images,n_single_detections,n_extra_detections = run_classifier_recursively(path=absPath,use_visual_output=use_visual_output,classifier=classifier,n_images=n_images,n_single_detections=n_single_detections,n_extra_detections=n_extra_detections,classifier_arg=classifier_arg)
               # adding the paths to the list that got traversed
                        donePaths.append(absPath)

    if n_images:
        positives = n_single_detections + n_extra_detections
        pos_rate = float(positives)/n_images
        neg_rate = float(n_images-positives)/n_images
        print('n_images:'+str(n_images)+' n_extra:'+str(n_extra_detections)+' n_detections:'+str(n_single_detections))
        print('pos rate:'+str(pos_rate)+' neg rate:'+str(neg_rate))
        return n_images,n_single_detections,n_extra_detections

    else:
        return 0,0,0

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


def show_rects(abs_path,faces,save_figs=True):
    img_arr = cv2.imread(abs_path)
    if len(faces):
        for rect in faces:
            print('rect:'+str(rect))
            cv2.rectangle(img_arr,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),[0,0,250],2)
    cv2.imshow('candidate',img_arr)
    newname  = abs_path.replace('.jpg','.cascaderects.jpg')
    cv2.imwrite(newname,img_arr)
    cv2.waitKey(10)

def depth_of_subdir_of_calling_function():
    '''
    this finds the depth of subdirectory in which the caller resides
    :return:
    '''
    path = os.getcwd()
 #   print('path:'+str(path))
    p2 = path.split('trendi_guru_modules')
  #  print('path split on trendigurumodules:'+str(p2))
    if len(p2) < 2:
        print('not in trendi_guru_modules')
    secondhalf = p2[1]
 #   print('secondhalf:'+str(secondhalf))
    cur = secondhalf.split('/')
 #   print('cur:'+str(cur))
    if len(cur) > 1:
        in_subdir_of_trendi_guru_modules = True
    return len(cur)-1

if __name__ == "__main__":

#    direct('.')
#    pos,neg = run_classifier_on_dir_of_dirs('/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/classifier_stuff/images/llamas')

#    n,singles,multiples = run_classifier_recursively('images/many_faces',use_visual_output=True,classifier=background_removal.find_face_cascade)
#    n,singles,multiples = run_classifier_recursively('images/many_faces',use_visual_output=True)
#    print('n:{0} single:{1} multiple:{2}'.format(n,singles,multiples))
#    raw_input('enter to continue')

    n,singles,multiples = run_classifier_recursively('images/many_faces',use_visual_output=True)
    print('n:{0} single:{1} multiple:{2}'.format(n,singles,multiples))
    raw_input('enter to continue')

    n,singles,multiples = run_classifier_recursively('/home/developer/python-packages/trendi_guru_modules/images/female_faces')
    print('n:{0} single:{1} multiple:{2}'.format(n,singles,multiples))
    raw_input('enter to continue')

    n,singles,multiples = run_classifier_recursively('/home/developer/python-packages/trendi_guru_modules/classifier_stuff/images/llamas')
    print('n:{0} single:{1} multiple:{2}'.format(n,singles,multiples))
    raw_input('enter to continue')

    n,singles,multiples = run_classifier_recursively('/home/developer/python-packages/trendi_guru_modules/classifier_stuff/images/monkeys')
    print('n:{0} single:{1} multiple:{2}'.format(n,singles,multiples))
    raw_input('enter to continue')

    n,singles,multiples = run_classifier_recursively('/home/developer/python-packages/trendi_guru_modules/classifier_stuff/images/male_faces')
    print('n:{0} single:{1} multiple:{2}'.format(n,singles,multiples))
    raw_input('enter to continue')

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

