'''
generally for reading db's having bb's
pascal voc
http://host.robots.ox.ac.uk/pascal/VOC/databases.html#VOC2005_2

'''

__author__ = 'jeremy'
import os
import cv2
import sys
import re
import pdb
import csv
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import json


from trendi import Utils
from trendi.classifier_stuff.caffe_nns import create_nn_imagelsts
from trendi.utils import imutils
from trendi import constants


def read_kitti(dir='/data/jeremy/image_dbs/hls/kitti/data_object_label_2',visual_output=True):
    '''
    reads data at http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/
    which has a file for each image, filenames 000000.txt, 000001.txt etc, each file has a line like:
    Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
    in format:
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
    1    alpha        Observation angle of object, ranging [-pi..pi]
    4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
    3    dimensions   3D object dimensions: height, width, length (in meters)
    3    location     3D object location x,y,z in camera coordinates (in meters)
    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
    :param dir:
    :return:
    '''

    files = os.listdir(dir)
    files.sort()
    for f in files:
    #    filename = os.path.join(dir,'%06d.txt'%i)
        if not os.path.exists(f):
            print('{} not found'.format(f))
        else:
            with open(f,'r' ) as fp:
                line = fp.read()
                print(line)
                try:
                    type,truncated,occluded,x1,y1,x2,y2,h,w,l,x,y,z,ry,score = line.split()
                except:
                    print("error:", sys.exc_info()[0])
                print('{} {} x1 {} y1 {} x2 {} y2 {}'.format(f,type,x1,y1,x2,y2))


def read_rmptfmp_write_yolo(images_dir='/data/jeremy/image_dbs/hls/data.vision.ee.ethz.ch',gt_file='refined.idl',class_no=0,visual_output=False,label_destination='labels'):
    '''
    reads from gt for dataset from https://data.vision.ee.ethz.ch/cvl/aess/dataset/  (pedestrians only)
    '"left/image_00000001.png": (212, 204, 232, 261):-1, (223, 181, 259, 285):-1, (293, 151, 354, 325):-1, (452, 208, 479, 276):-1, (255, 219, 268, 249):-1, (280, 219, 291, 249):-1, (267, 246, 279, 216):-1, (600, 247, 584, 210):-1;'
    writes to yolo format
    '''

    # Define the codec and create VideoWriter object
    # not necessary fot function , just wanted to track boxes
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#    pdb.set_trace()
    with open(os.path.join(images_dir,gt_file),'r') as fp:
        lines = fp.readlines()
        for line in lines:
            print line
            elements = re.findall(r"[-\w']+",line)
            print elements
        #    elements = line.split
            imgname = line.split()[0].replace('"','').replace(':','').replace('\n','')#.replace('.png','_0.png')
        #    print('img name '+str(imgname))
            imgname = os.path.basename(imgname) #ignore dir referred to in gt file and use mine
            if imgname[-6:] != '_0.png':
                print('imgname {} has no _0 at end'.format(imgname))
                imgname = imgname.replace('.png','_0.png')
            fullpath=os.path.join(images_dir,imgname)
            if not os.path.isfile(fullpath):
                print('couldnt find {}'.format(fullpath))
                continue
            print('reading {}'.format(fullpath))
            img_arr = cv2.imread(fullpath)
            img_dims = (img_arr.shape[1],img_arr.shape[0]) #widthxheight
            png_element_index = elements.index('png')
            bb_list_xywh = []
            ind = png_element_index+1
            n_bb=0
            while ind<len(elements):
                x1=int(elements[ind])
                if x1 == -1:
                    ind=ind+1
                    x1=int(elements[ind])
                y1=int(elements[ind+1])
                x2=int(elements[ind+2])
                y2=int(elements[ind+3])
                ind = ind+4
                if y2 == -1:
                    print('XXX warning, got a -1 XXX')
                n_bb += 1
                bb = Utils.fix_bb_x1y1x2y2([x1,y1,x2,y2])
                bb_xywh = [bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]]
                bb_list_xywh.append(bb_xywh)
                print('ind {} x1 {} y1 {} x2 {} y2 {} bbxywh {}'.format(ind,x1,y1,x2,y2,bb_xywh))
                if visual_output:
                    cv2.rectangle(img_arr,(x1,y1),(x2,y2),color=[100,255,100],thickness=2)
            print('{} bounding boxes for this image (png {} len {} '.format(n_bb,png_element_index,len(elements)))
            print('sending {} for writing'.format(bb_list_xywh))
            write_yolo_labels(fullpath,bb_list_xywh,class_no,img_dims)
            if visual_output:
                cv2.imshow('img',img_arr)
                cv2.waitKey(0)
 #           out.write(img_arr)
 #       out.release()
    if visual_output:
        cv2.destroyAllWindows()


def write_yolo_labels(img_path,bb_list_xywh,class_number,image_dims,destination_dir=None,overwrite=True):
    '''
    output : for yolo - https://pjreddie.com/darknet/yolo/
    Darknet wants a .txt file for each image with a line for each ground truth object in the image that looks like:
    <object-class> <x> <y> <width> <height>
    where those are percentages...
    it looks like yolo makes an assumption abt where images and label files are, namely in parallel dirs. named:
    JPEGImages  labels
    and a train.txt file pointing to just the images - and the label files are same names with .txt instead of .jpg
    :param img_path:
    :param bb_xywh:
    :param class_number:
    :param destination_dir:
    :return:
    '''
    if destination_dir is None:
        destination_dir = Utils.parent_dir(os.path.basename(img_path))
        destination_dir = os.path.join(destination_dir,'labels')
        Utils.ensure_dir(destination_dir)
    img_basename = os.path.basename(img_path)
    img_basename = img_basename.replace('.jpg','.txt').replace('.png','.txt').replace('.bmp','.txt')
    destination_path=os.path.join(destination_dir,img_basename)
    if overwrite:
        write_mode = 'w'
    else:
        write_mode = 'a'
    with open(destination_path,write_mode) as fp:
        for bb_xywh in bb_list_xywh:
            x_center = bb_xywh[0]+bb_xywh[2]/2.0
            y_center = bb_xywh[1]+bb_xywh[3]/2.0
            x_p = float(x_center)/image_dims[0]
            y_p = float(y_center)/image_dims[1]
            w_p = float(bb_xywh[2])/image_dims[0]
            h_p = float(bb_xywh[3])/image_dims[1]
            line = str(class_number)+' '+str(round(x_p,4))+' '+str(round(y_p,4))+' '+str(round(w_p,4))+' '+str(round(h_p,4))+'\n'
            print('writing "{}" to {}'.format(line[:-1],destination_path))
            fp.write(line)
    fp.close()
#    if not os.exists(destination_path):
#        Utils.ensure_file(destination_path)

def write_yolo_trainfile(image_dir,trainfile='train.txt',filter='.png',split_to_test_and_train=0.05,check_for_bbfiles=True,bb_dir=None):
    '''
    this is just a list of full paths to the training images. the labels apparently need to be in parallel dir(s) called 'labels'
    note this appends to trainfile , doesnt overwrite , to facilitate building up from multiple sources
    :param dir:
    :param trainfile:
    :return:
    '''
    files = [os.path.join(image_dir,f) for f in os.listdir(image_dir) if filter in f]
    print('{} files w filter {} in {}'.format(len(files),filter,image_dir))
    if check_for_bbfiles:
        if bb_dir == None:
            bb_dir = os.path.join(Utils.parent_dir(image_dir),os.path.basename(image_dir)+'labels')
        print('checkin for bbs in '+bb_dir)
    if len(files) == 0:
        print('no files fitting {} in {}, stopping'.format(filter,image_dir))
        return
    count = 0
    with open(trainfile,'a+') as fp:
        for f in files:
            if check_for_bbfiles:
                bbfile = os.path.basename(f).replace(filter,'.txt')
                bbpath = os.path.join(bb_dir,bbfile)
                if os.path.exists(bbpath):
                    fp.write(f+'\n')
                    count +=1
                else:
                    print('bbfile {} describing {} not found'.format(bbpath,f))
            else:
                fp.write(f+'\n')
                count += 1
    print('wrote {} files to {}'.format(count,trainfile))
    if split_to_test_and_train is not None:
        create_nn_imagelsts.split_to_trainfile_and_testfile(trainfile,fraction=split_to_test_and_train)

def read_yolo_bbs(txt_file,img_file):
    '''
    format is
    <object-class> <x> <y> <width> <height>
    where x,y,w,h are relative to image width, height.  It looks like x,y are bb center, not topleft corner - see voc_label.py in .convert(size,box) func
    :param txt_file:
    :return:
    '''
#    img_file = txt_file.replace('.txt','.png')
    img_arr = cv2.imread(img_file)
    if img_arr is None:
        print('problem reading {}'.format(img_file))
    image_h,image_w = img_arr.shape[0:2]
    with open(txt_file,'r') as fp:
        lines = fp.readlines()
        print('{} bbs found'.format(len(lines)))
        for line in lines:
            object_class,x,y,w,h = line.split()
            x_p=float(x)
            y_p=float(y)
            w_p=float(w)
            h_p=float(h)
            x_center = int(x_p*image_w)
            y_center = int(y_p*image_h)
            w = int(w_p*image_w)
            h = int(h_p*image_h)
            x1 = x_center-w/2
            x2 = x_center+w/2
            y1 = y_center-h/2
            y2 = y_center+h/2
            print('class {} x_c {} y_c {} w {} h {} x x1 {} y1 {} x2 {} y2 {}'.format(object_class,x_center,y_center,w,h,x1,y1,x2,y2))
            cv2.rectangle(img_arr,(x1,y1),(x2,y2),color=[100,255,100],thickness=2)
            cv2.imshow('img',img_arr)
        cv2.waitKey(0)

def read_many_yolo_bbs(imagedir='/data/jeremy/image_dbs/hls/data.vision.ee.ethz.ch/left/',labeldir=None,img_filter='.png'):
    if labeldir is None:
        labeldir = os.path.join(Utils.parent_dir(imagedir),'labels')
    imgfiles = [f for f in os.listdir(imagedir) if img_filter in f]
    imgfiles = sorted(imgfiles)
    print('found {} files in {}, label dir {}'.format(len(imgfiles),imagedir,labeldir))
    for f in imgfiles:
        bb_path = os.path.join(labeldir,f).replace(img_filter,'.txt')
        if not os.path.isfile(bb_path):
            print('{} not found '.format(bb_path))
            continue
        image_path = os.path.join(imagedir,f)
        read_yolo_bbs(bb_path,image_path)

def read_pascal_xml_write_yolo(dir='/media/jeremy/9FBD-1B00/hls_potential/voc2007/VOCdevkit/VOC2007',annotation_folder='Annotations',img_folder='JPEGImages',
                               annotation_filter='.xml'):
    '''
    nondestructive - if there are already label files these get added to not overwritten
    :param dir:
    :param annotation_folder:
    :param img_folder:
    :param annotation_filter:
    :return:
    '''
#    classes = [ 'person','hat','backpack','bag','person_wearing_red_shirt','person_wearing_blue_shirt',
#                       'car','bus','truck','unattended_bag', 'bicycle',  'motorbike']

    classes = constants.hls_yolo_categories
    annotation_dir = os.path.join(dir,annotation_folder)
    img_dir = os.path.join(dir,img_folder)
    annotation_files = [os.path.join(annotation_dir,f) for f in os.listdir(annotation_dir) if annotation_filter in f]
    listfilename = os.path.join(dir,'filelist.txt')
    list_file = open(listfilename, 'w')
    for annotation_file in annotation_files:
        success = convert_pascal_xml_annotation(annotation_file,classes)
        if success:
            print('found relevant class(es)')
            filenumber = os.path.basename(annotation_file).replace('.xml','')
            jpgpath = os.path.join(img_dir,str(filenumber)+'.jpg')
            list_file.write(jpgpath+'\n')

def convert_pascal_xml_annotation(in_file,classes,labeldir=None):
    filenumber = os.path.basename(in_file).replace('.xml','')
#    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    if labeldir==None:
        parent_dir = Utils.parent_dir(os.path.dirname(in_file))
        labeldir = os.path.join(parent_dir,'labels')
        Utils.ensure_dir(labeldir)
    out_filename = os.path.join(labeldir, filenumber+'.txt')
    print('in {} out {}'.format(in_file,out_filename))
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    success=False
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_x1x2y1y2_to_yolo((w,h), b)
        out_file = open(out_filename, 'a+')
        os.chmod(out_filename, 0o666)
        out_file.write(str(cls_id) + " " + " ".join([str(round(a,4)) for a in bb]) + '\n')
 #       os.chmod(out_filename, 0o777)
        success = True
    return(success)

def read_pascal_txt_write_yolo(dir='/media/jeremy/9FBD-1B00/hls_potential/voc2005_1/',
                               annotation_folder='all_relevant_annotations',img_folder='all_relevant_images',
                               annotation_filter='.txt',image_filter='.png',yolo_annotation_dir='labels'):
    '''
    nondestructive - if there are already label files these get added to not overwritten
    :param dir:
    :param annotation_folder:
    :param img_folder:
    :param annotation_filter:
    :return:
    '''
#    classes = [ 'person','hat','backpack','bag','person_wearing_red_shirt','person_wearing_blue_shirt',
#                       'car','bus','truck','unattended_bag', 'bicycle',  'motorbike']

    classes = constants.hls_yolo_categories

    annotation_dir = os.path.join(dir,annotation_folder)
    img_dir = os.path.join(dir,img_folder)
    annotation_files = [os.path.join(annotation_dir,f) for f in os.listdir(annotation_dir) if annotation_filter in f]
    listfilename = os.path.join(dir,'filelist.txt')
    list_file = open(listfilename, 'w')
    yolo_annotation_path = os.path.join(dir,yolo_annotation_dir)
    Utils.ensure_dir(yolo_annotation_path)
    for annotation_file in annotation_files:

        out_filename=os.path.join(yolo_annotation_path,os.path.basename(annotation_file))
        print('outfile'+out_filename)
        success = convert_pascal_txt_annotation(annotation_file,classes,out_filename)
        if success:
            print('found relevant class(es)')
            filename = os.path.basename(annotation_file).replace(annotation_filter,'')
            img_dir =  os.path.join(dir,img_folder)
            imgpath = os.path.join(img_dir,str(filename)+image_filter)
            list_file.write(imgpath+'\n')

def convert_pascal_txt_annotation(in_file,classes,out_filename):
    print('in {} out {}'.format(in_file,out_filename))
    with open(in_file,'r') as fp:
        lines = fp.readlines()
    for i in range(len(lines)):
        if 'Image filename' in lines[i]:
            imfile=lines[i].split()[3]
            print('imfile:'+imfile)
            # path = Utils.parent_dir(os.path.basename(in_file))
            # if path.split('/')[-1] != 'Annotations':
            #     path = Utils.parent_dir(path)
            # print('path to annotation:'+str(path))
            # img_path = os.path.join(path,imfile)
            # print('path to img:'+str(img_path))
            # img_arr = cv2.imread(img_path)
        if 'Image size' in lines[i]:
            nums = re.findall('\d+', lines[i])
            print(lines[i])
            print('nums'+str(nums))
            w = int(nums[0])
            h = int(nums[1])
            print('h {} w {}'.format(h,w))
        if '# Details' in lines[i] :
            object = lines[i].split()[5].replace('(','').replace(')','').replace('"','')
            nums = re.findall('\d+', lines[i+2])
            print('obj {} nums {}'.format(object,nums))
            success=False
            cls_id = tg_class_from_pascal_class(object,classes)
            if cls_id is not None:
                print('class index '+str(cls_id)+' '+classes[cls_id])
                success=True
            if not success:
                print('NO RELEVANT CLASS FOUND')
                continue
            b = (int(nums[1]), int(nums[3]), int(nums[2]), int(nums[4])) #file has xmin ymin xmax ymax
            print('bb_x1x2y1y2:'+str(b))
            bb = convert_x1x2y1y2_to_yolo((w,h), b)
            print('bb_yolo'+str(bb))
            if os.path.exists(out_filename):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not

            out_file = open(out_filename, append_write)
  #          os.chmod(out_filename, 0o666) #
            out_file.write(str(cls_id) + " " + " ".join([str(round(a,4)) for a in bb]) + '\n')
#       os.chmod(out_filename, 0o777)
        success = True
    return(success)


def tg_class_from_pascal_class(pascal_class,tg_classes):
#hls_yolo_categories = [ 'person','hat','backpack','bag','person_wearing_red_shirt','person_wearing_blue_shirt',
#                       'car','bus','truck','unattended_bag', 'bicycle',  'motorbike']

    conversions = {'bike':'bicycle',
                   'motorcycle':'motorbike'}  #things that have names different than tg names
                                            #(forced to do this since e.g. bike and bicycle are both used in VOC)
    for tg_class in tg_classes:
        if tg_class in pascal_class:
            tg_ind = tg_classes.index(tg_class)
            return tg_ind
    for pascal,tg in conversions.iteritems():
        if pascal in pascal_class:
            tg_ind = tg_classes.index(tg)
            return tg_ind
    return None

def write_yolo_from_tgdict(tg_dict,label_dir=None,classes=constants.hls_yolo_categories):
    '''
    input- dict in 'tg format' which is like this
       {'filename':'image423.jpg','annotations':[{'object':'person','bbox_xywh':[x,y,w,h]},{'object':'person','bbox_xywh':[x,y,w,h],'sId':104}],
    {'filename':'image423.jpg','annotations':[{'object':'person','bbox_xywh':[x,y,w,h]},{'object':'person','bbox_xywh':[x,y,w,h],'sId',105}
    That json can then be used to generate yolo or frcnn training files
    output : for yolo - https://pjreddie.com/darknet/yolo/
    Darknet wants a .txt file for each image with a line for each ground truth object in the image that looks like:
    <object-class> <x> <y> <width> <height>
    where those are percentages...
    it looks like yolo makes an assumption abt where images and label files are, namely in parallel dirs named [whatever]images and [whatever]labels:
    e.g. JPEGImages  labels
    and a train.txt file pointing to just the images - the label files are same names with .txt instead of .jpg
    :param img_path:
    :param bb_xywh:
    :param class_number:
    :param destination_dir:
    :return:
    '''
    img_filename = tg_dict['filename']
    annotations = tg_dict['annotations']
    sid = None
    if 'sid' in tg_dict:
        sid = tg_dict['sId']
    dims = tg_dict['dimensions_h_w_c']
    im_h,im_w=(dims[0],dims[1])
    print('file {}\nannotations {}'.format(img_filename,annotations))
    if label_dir is None:
        img_parent = Utils.parent_dir(os.path.dirname(img_filename))
        img_diralone = os.path.dirname(img_filename).split('/')[-1]
        label_diralone = img_diralone+'labels'
        label_dir= os.path.join(img_parent,label_diralone)
        Utils.ensure_dir(label_dir)
     #   label_dir = os.path.join(img_parent,label_ext)
        print('img parent {} labeldir {} imgalone {} lblalone {} '.format(img_parent,label_dir,img_diralone,label_diralone))
    label_name = os.path.basename(img_filename).replace('.png','.txt').replace('.jpg','.txt')
    label_path = os.path.join(label_dir,label_name)
    print('writing to '+str(label_path))
    with open(label_path,'w') as fp:
        for annotation in annotations:
            bb_xywh = annotation['bbox_xywh']
            bb_yolo = imutils.xywh_to_yolo(bb_xywh,(im_w,im_h))
            print('dims {} bbxywh {} bbyolo {}'.format((im_w,im_h),bb_xywh,bb_yolo))
            object = annotation['object']
            class_number = classes.index(object)
            line = str(class_number)+' '+str(bb_yolo[0])+' '+str(bb_yolo[1])+' '+str(bb_yolo[2])+' '+str(bb_yolo[3])+'\n'
            fp.write(line)
        fp.close()

def autti_txt_to_yolo(autti_txt='/media/jeremy/9FBD-1B00/image_dbs/hls/object-dataset/labels.csv'):
    #to deal with driving file from autti
#   wget  http://bit.ly/udacity-annotations-autti
    all_annotations = txt_to_tgdict(txtfile=autti_txt,image_dir=None,parsemethod=parse_autti)
    for tg_dict in all_annotations:
        write_yolo_from_tgdict(tg_dict)

    json_name = autti_txt.replace('.csv','.json')
    inspect_json(json_name)

def udacity_csv_to_yolo(udacity_csv='/media/jeremy/9FBD-1B00/image_dbs/hls/object-detection-crowdai/labels.csv'):
# to deal with driving  file from udacity -
#  wget http://bit.ly/udacity-annoations-crowdai

    all_annotations = csv_to_tgdict(udacity_csv=udacity_csv,parsemethod=parse_udacity)
    for tg_dict in all_annotations:
        write_yolo_from_tgdict(tg_dict)

    json_name = udacity_csv.replace('.csv','.json')
    inspect_json(json_name)

def parse_udacity(row):
    xmin=int(row['xmin'])
    xmax=int(row['ymin'])
    ymin=int(row['xmax'])
    ymax=int(row['ymax'])
    frame=row['Frame']  #aka filename
    label=row['Label']
    label=label.lower()
    preview_url=row['Preview URL']
    tg_object=convert_udacity_label_to_tg(label)
    if tg_object is None:
        #label didnt get xlated so its something we dont care about e.g streetlight
        print('object {} is not of interest'.format(label))
    return xmin,xmax,ymin,ymax,frame,tg_object

def parse_autti(row,delimiter=' '):
    #these parse guys should also have the translator (whatever classes into tg classes
#autti looks like this
#   178019968680240537.jpg 888 498 910 532 0 "trafficLight" "Red"
#   1478019969186707568.jpg 404 560 540 650 0 "car"
    elements = row.split(delimiter)
    filename=elements[0]
    xmin=int(elements[1])
    ymin=int(elements[2])
    xmax=int(elements[3])
    ymax=int(elements[4])
    #something i'm ignoring in row[5]
    label=elements[6].replace('"','').replace("'","").replace('\n','').replace('\t','')
    label=label.lower()

    assert(xmin<xmax)
    assert(ymin<ymax)
    tg_object=convert_udacity_label_to_tg(label)
    if tg_object is None:
        #label didnt get xlated so its something we dont care about e.g streetlight
        print('object {} is not of interest'.format(label))
    return xmin,xmax,ymin,ymax,filename,tg_object

def csv_to_tgdict(udacity_csv='/media/jeremy/9FBD-1B00/image_dbs/hls/object-dataset/labels.csv',image_dir=None,classes=constants.hls_yolo_categories,visual_output=False,manual_verification=False,jsonfile=None,parsemethod=parse_udacity,delimiter='\t',readmode='r'):
    '''
    read udaicty csv to grab files here
    https://github.com/udacity/self-driving-car/tree/master/annotations

    pedestrians, cars, trucks (and trafficlights in second one)
    udacity file looks like:
    xmin,ymin,xmax,ymax,Frame,Label,Preview URL
    785,533,905,644,1479498371963069978.jpg,Car,http://crowdai.com/images/Wwj-gorOCisE7uxA/visualize
    create the 'usual' tg dict for bb's , also write to json while we're at it
    [ {
        "dimensions_h_w_c": [360,640,3],
        "filename": "/data/olympics/olympics/9908661.jpg"
        "annotations": [
            {
               "bbox_xywh": [89, 118, 64,44 ],
                "object": "car"
            }
        ],   }, ...

    :param udacity_csv:
    :param label_dir:
    :param classes:
    :return:
    '''
#todo this can be combined with the txt_to_tgdict probably, maybe usin csv.reader instead of csv.dictread
#    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#...     for row in spamreader:
#...         print ', '.join(row)

    all_annotations = []
    if image_dir is None:
        image_dir = os.path.dirname(udacity_csv)
    print('opening udacity csv file {} '.format(udacity_csv))
  #  with open(udacity_csv, newline='') as file:
    with open(udacity_csv,readmode) as file:
 #   with open('eggs.csv', newline='') as csvfile:
##        reader = csv.DictReader(file,delimiter=delimiter, quotechar='|')
        reader = csv.DictReader(file)
        n_rows = 0
        max_annotations=10**10
        for row in reader:
            n_rows += 1
            print('row'+str(row))
            try:
                xmin,xmax,ymin,ymax,filename,label=parsemethod(row)
                print('file {} xmin {} ymin {} xmax {} ymax {} object {}'.format(filename,xmin,ymin,xmax,ymax,label))
            except:
                print('trouble getting row '+str(row))
                continue
            try:
                assert(xmax>xmin)
                assert(ymax>ymin)
            except:
                print('problem with order of x/y min/max')
                print('xmin {} ymin {} xmax {} ymax {} '.format(xmin,ymin,xmax,ymax))
                xmint=min(xmin,xmax)
                xmax=max(xmin,xmax)
                xmin=xmint
                ymint=min(ymin,ymax)
                ymax=max(ymin,ymax)
                ymin=ymint
            bb = [xmin,ymin,xmax-xmin,ymax-ymin]  #xywh
            if image_dir is not None:
                full_name = os.path.join(image_dir,filename)
            else:
                full_name = filename
            im = cv2.imread(full_name)
            if im is None:
                print('couldnt open '+full_name)
                continue
            im_h,im_w=im.shape[0:2]

            annotation_dict = {}
            annotation_dict['filename']=full_name
            annotation_dict['annotations']=[]
            annotation_dict['dimensions_h_w_c'] = im.shape
            #check if file has already been seen and a dict started, if so use that instead
            file_already_in_json = False
            #this is prob a stupid slow way to check
            for a in all_annotations:
                if a['filename'] == full_name:
                    annotation_dict=a
                    file_already_in_json = True
                    break
#            print('im_w {} im_h {} bb {} label {}'.format(im_w,im_h,bb,label))
            object_dict={}
            object_dict['bbox_xywh'] = bb
            object_dict['object']=label

            if visual_output or manual_verification:
                im = imutils.bb_with_text(im,bb,label)
                magnify = 1
                im = cv2.resize(im,(int(magnify*im_w),int(magnify*im_h)))
                cv2.imshow('full',im)
                if not manual_verification:
                    cv2.waitKey(5)

                else:
                    print('(a)ccept , any other key to not accept')
                    k=cv2.waitKey(0)
                    if k == ord('a'):
                        annotation_dict['annotations'].append(object_dict)
                    else:
                        continue #dont add bb to list, go to next csv line
            if not manual_verification:
                annotation_dict['annotations'].append(object_dict)
           # print('annotation dict:'+str(annotation_dict))
            if not file_already_in_json: #add new file to all_annotations
                all_annotations.append(annotation_dict)
            else:  #update current annotation with new bb
                for a in all_annotations:
                    if a['filename'] == full_name:
                        a=annotation_dict
     #       print('annotation dict:'+str(annotation_dict))
            print('# files:'+str(len(all_annotations)))
            if len(all_annotations)>max_annotations:
                break #  for debugging, these files are ginormous
           # raw_input('ret to cont')

    if jsonfile == None:
        jsonfile = udacity_csv.replace('.csv','.json')
    with open(jsonfile,'w') as fp:
        json.dump(all_annotations,fp,indent=4)
        fp.close()

    return all_annotations

def txt_to_tgdict(txtfile='/media/jeremy/9FBD-1B00/image_dbs/hls/object-dataset/labels.csv',image_dir=None,classes=constants.hls_yolo_categories,visual_output=False,manual_verification=False,jsonfile=None,parsemethod=parse_autti,wait=1):
    '''
    read udaicty csv to grab files here
    https://github.com/udacity/self-driving-car/tree/master/annotations
    pedestrians, cars, trucks (and trafficlights in second one)
    udacity file looks like:
    xmin,ymin,xmax,ymax,Frame,Label,Preview URL
    785,533,905,644,1479498371963069978.jpg,Car,http://crowdai.com/images/Wwj-gorOCisE7uxA/visualize
    create the 'usual' tg dict for bb's , also write to json while we're at it
    [ {
        "dimensions_h_w_c": [360,640,3],
        "filename": "/data/olympics/olympics/9908661.jpg"
        "annotations": [
            {
               "bbox_xywh": [89, 118, 64,44 ],
                "object": "car"
            }
        ],   }, ...

    :param udacity_csv:
    :param label_dir:
    :param classes:
    :return:
    '''

    all_annotations = []
    if image_dir is None:
        image_dir = os.path.dirname(txtfile)
    print('opening udacity csv file {} '.format(txtfile))
    with open(txtfile, "r") as file:
        lines = file.readlines()
        for row in lines:
#            print(row)
            try:
                xmin,xmax,ymin,ymax,filename,label=parsemethod(row)
                print('file {} xmin {} ymin {} xmax {} ymax {} object {}'.format(filename,xmin,ymin,xmax,ymax,label))
                if label is None:
                    continue
            except:
                print('trouble getting row '+str(row))
                continue

            try:
                assert(xmax>xmin)
                assert(ymax>ymin)
            except:
                print('problem with order of x/y min/max')
                print('xmin {} ymin {} xmax {} ymax {} '.format(xmin,ymin,xmax,ymax))
                xmint=min(xmin,xmax)
                xmax=max(xmin,xmax)
                xmin=xmint
                ymint=min(ymin,ymax)
                ymax=max(ymin,ymax)
                ymin=ymint
            if image_dir is not None:
                full_name = os.path.join(image_dir,filename)
            else:
                full_name = filename

            im = cv2.imread(full_name)
            if im is None:
                print('couldnt open '+full_name)
                continue
            im_h,im_w=im.shape[0:2]

            annotation_dict = {}
            bb = [xmin,ymin,xmax-xmin,ymax-ymin]  #xywh

            annotation_dict['filename']=full_name
            annotation_dict['annotations']=[]
            annotation_dict['dimensions_h_w_c'] = im.shape
            #check if file has already been seen and a dict started, if so use that instead
            file_already_in_json = False
            #this is prob a stupid slow way to check
            for a in all_annotations:
                if a['filename'] == full_name:
                    annotation_dict=a
                    file_already_in_json = True
                    break
            object_dict={}
            object_dict['bbox_xywh'] = bb
            object_dict['object']=label

            if visual_output or manual_verification:
                im = imutils.bb_with_text(im,bb,label)
                magnify = 1
                im = cv2.resize(im,(int(magnify*im_w),int(magnify*im_h)))
                cv2.imshow('full',im)
                if not manual_verification:
                    cv2.waitKey(wait)

                else:
                    print('(a)ccept , any other key to not accept')
                    k=cv2.waitKey(0)
                    if k == ord('a'):
                        annotation_dict['annotations'].append(object_dict)
                    else:
                        continue #dont add bb to list, go to next csv line
            if not manual_verification:
                annotation_dict['annotations'].append(object_dict)
           # print('annotation dict:'+str(annotation_dict))
            if not file_already_in_json: #add new file to all_annotations
                all_annotations.append(annotation_dict)
            else:  #update current annotation with new bb
                for a in all_annotations:
                    if a['filename'] == full_name:
                        a=annotation_dict
     #       print('annotation dict:'+str(annotation_dict))
            print('# files:'+str(len(all_annotations)))
           # raw_input('ret to cont')

    if jsonfile == None:
        jsonfile = txtfile.replace('.csv','.json').replace('.txt','.json')
    with open(jsonfile,'w') as fp:
        json.dump(all_annotations,fp,indent=4)
        fp.close()

    return all_annotations

def convert_udacity_label_to_tg(udacity_label):
#    hls_yolo_categories = ['person','person_wearing_hat','person_wearing_backpack','person_holding_bag',
#                       'man_with_red_shirt','man_with_blue_shirt',
#                       'car','van','truck','unattended_bag']
#udacity: Car Truck Pedestrian

    conversions = {'pedestrian':'person',
                   'car':'car',
                   'truck':'truck'}
    if not udacity_label in conversions:
        print('!!!!!!!!!! did not find {} in conversions from udacity to tg cats !!!!!!!!'.format(udacity_label))
#        raw_input('!!')
        return(None)
    tg_description = conversions[udacity_label]
    return(tg_description)

def convert_x1x2y1y2_to_yolo(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def inspect_yolo_annotations(dir='/media/jeremy/9FBD-1B00/hls_potential/voc2007/VOCdevkit/VOC2007',yolo_annotation_folder='labels',img_folder='JPEGImages',
                               annotation_filter='.txt',image_filter='.jpg'):
    #https://www.youtube.com/watch?v=c-vhrv-1Ctg   jinjer
    annotation_dir = os.path.join(dir,yolo_annotation_folder)
    img_dir = os.path.join(dir,img_folder)
    annotation_files = [os.path.join(annotation_dir,f) for f in os.listdir(annotation_dir) if annotation_filter in f]
    classes = constants.hls_yolo_categories
    print('inspecting yolo annotations in '+dir)
    for f in annotation_files:
        print('trying '+f)
        annotation_base = os.path.basename(f)
        imgfile = annotation_base.replace(annotation_filter,image_filter)
        img_path = os.path.join(img_dir,imgfile)
        img_arr = cv2.imread(img_path)
        if img_arr is None:
            print('coulndt get '+img_path)
        h,w = img_arr.shape[0:2]
        with open(f,'r') as fp:
            lines = fp.readlines()
            for line in lines:
                print(line)
                object_class,bb0,bb1,bb2,bb3 = line.split()
                bb_xywh = imutils.yolo_to_xywh([float(bb0),float(bb1),float(bb2),float(bb3)],(w,h))
                classname = classes[int(object_class)]
                print('class {} bb_xywh {}'.format(classname,bb_xywh))
                cv2.rectangle(img_arr,(bb_xywh[0],bb_xywh[1]),(bb_xywh[0]+bb_xywh[2],bb_xywh[1]+bb_xywh[3]),color=[100,255,100],thickness=2)
                img_arr[bb_xywh[1]:bb_xywh[1]+20,bb_xywh[0]:bb_xywh[0]+bb_xywh[2]]=img_arr[bb_xywh[1]:bb_xywh[1]+20,bb_xywh[0]:bb_xywh[0]+bb_xywh[2]]/2+[100,50,100]
                cv2.putText(img_arr,classname,(bb_xywh[0]+5,bb_xywh[1]+20),cv2.FONT_HERSHEY_PLAIN, 1, [255,0,255])
                cv2.imshow('img',img_arr)
            cv2.waitKey(0)



def inspect_json(jsonfile='rio.json',visual_output=False,check_img_existence=True,movie=False):
    '''
        read file like:
        [{'filename':'image423.jpg','annotations':[{'object':'person','bbox_xywh':[x,y,w,h]},{'object':'person','bbox_xywh':[x,y,w,h],'sId':104}],
    {'filename':'image423.jpg','annotations':[{'object':'person','bbox_xywh':[x,y,w,h]},{'object':'person','bbox_xywh':[x,y,w,h],'sId',105} ,...]
    :param jsonfile:
    :return:
    '''
    #todo add visual inspect here
    object_counts = {}
    print('inspecting json annotations in '+jsonfile)
    with open(jsonfile,'r') as fp:
        annotation_list = json.load(fp)


# Define the codec and create VideoWriter object
    if movie:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
#        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


    for d in annotation_list:
#        print d
        filename = d['filename']
        annotations = d['annotations']
        sid = None
        if 'sId' in d:
            sid = d['sId']
        n_bbs = len(annotations)
        print('file {}\n{} annotations {}\nsid {}'.format(filename,n_bbs,annotations,sid))
        if check_img_existence:
            if not os.path.exists(filename):
                print('WARNNING could not find '+filename+' WARNING')
        if visual_output:
            img_arr = cv2.imread(filename)
            if img_arr is None:
                print('WARNNING could not read '+filename+' WARNING')

        for annotation in annotations:
            object = annotation['object']
            bb_xywh = annotation['bbox_xywh']
            if visual_output:
                imutils.bb_with_text(img_arr,bb_xywh,object)
            if not object in object_counts:
                object_counts[object] = 1
            else:
                object_counts[object] = object_counts[object] + 1
        if visual_output:
            cv2.imshow('out',img_arr)
            cv2.waitKey(0)
            if movie:
                out.write(img_arr)


    print('n annotated files {}'.format(len(annotation_list)))
    print('bb counts by category {}'.format(object_counts))
    if visual_output:
        cv2.destroyAllWindows()

    if movie:
        out.release()



if __name__ == "__main__":
    txt_to_tgdict()