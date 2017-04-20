'''
generally for reading db's having bb's

'''

__author__ = 'jeremy'
import os
import cv2
import sys
import re
import pdb

from trendi import Utils
from trendi.classifier_stuff.caffe_nns import create_nn_imagelsts

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
                imgname = imgname.replace('.png','_0.png')
            fullpath=os.path.join(images_dir,imgname)
            if not os.path.isfile(fullpath):
                print('couldnt find {}'.format(fullpath))
                continue
            print('reading {}'.format(fullpath))
            img_arr = cv2.imread(fullpath)
            img_dims = (img_arr.shape[1],img_arr.shape[0]) #widthxheight
            png_element_index = elements.index('png')
            n_bb = (len(elements) - png_element_index)/5  #3 elements till first bb, five elem per bb
            print('{} bounding boxes for this image (png {} len {} '.format(n_bb,png_element_index,len(elements)))
            bb_list_xywh = []
            for i in range(int(n_bb)):
                ind = i*5+png_element_index+1
                x1=int(elements[ind])
                y1=int(elements[ind+1])
                x2=int(elements[ind+2])
                y2=int(elements[ind+3])
                bb = Utils.fix_bb_x1y1x2y2([x1,y1,x2,y2])
                bb_xywh = [bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]]
                bb_list_xywh.append(bb_xywh)
                print('ind {} x1 {} y1 {} x2 {} y2 {} bbxywh {}'.format(ind,x1,y1,x2,y2,bb_xywh))
                if visual_output:
                    cv2.rectangle(img_arr,(x1,y1),(x2,y2),color=[100,255,100],thickness=2)
                write_yolo_labels(fullpath,bb_list_xywh,class_no,img_dims)
            if visual_output:
                cv2.imshow('img',img_arr)
                cv2.waitKey(0)
 #           out.write(img_arr)
 #       out.release()
    if visual_output:
        cv2.destroyAllWindows()

def write_yolo_labels(img_path,bb_list_xywh,class_number,image_dims,destination_dir=None):
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
        destination_dir = Utils.parent_dir(img_path)
        destination_dir = os.path.join(destination_dir,'labels')
        Utils.ensure_dir(destination_dir)
    img_basename = os.path.basename(img_path)
    img_basename = img_basename.replace('.jpg','.txt').replace('.png','.txt').replace('.bmp','.txt')
    destination_path=os.path.join(destination_dir,img_basename)
    with open(destination_path,'w+') as fp:
        for bb_xywh in bb_list_xywh:
            x_center = bb_xywh[0]+bb_xywh[2]/2.0
            y_center = bb_xywh[1]+bb_xywh[3]/2.0
            x_p = float(x_center)/image_dims[0]
            y_p = float(y_center)/image_dims[1]
            w_p = float(bb_xywh[2])/image_dims[0]
            h_p = float(bb_xywh[3])/image_dims[1]
            line = str(class_number)+' '+str(round(x_p,4))+' '+str(round(y_p,4))+' '+str(round(w_p,4))+' '+str(round(h_p,4))+'\n'
            print('writing "{}" to {}'.format(line,destination_path))
            fp.write(line)
    fp.close()
#    if not os.exists(destination_path):
#        Utils.ensure_file(destination_path)

def write_yolo_trainfile(dir,trainfile='train.txt',filter='.png',split_to_test_and_train=0.05):
    '''
    this is just a list of full paths to the training images. the labels apparently need to be in parallel dir(s) called 'labels'
    :param dir:
    :param trainfile:
    :return:
    '''
    files = [os.path.join(dir,f) for f in os.listdir(dir) if filter in f]
    print('{} files w filter {} in {}'.format(len(files),filter,dir))
    if len(files) == 0:
        print('no files fitting {} in {}, stopping'.format(filter,dir))
        return
    with open(trainfile,'w+') as fp:
        for f in files:
            fp.write(f+'\n')
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

if __name__ == "__main__":
    read_many_yolo_bbs(imagedir='/data/jeremy/image_dbs/hls/data.vision.ee.ethz.ch/JELMOLI/images')