__author__ = 'jeremy'

import os
import cv2

def read_kitti(dir='/data/jeremy/image_dbs/hls/kitti',visual_output=True):
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
    for i in range(len(files)):
        filename = os.path.join(dir,'%06d.txt'%i)
        if not os.path.exists(filename):
            print('{} not found'.format(filename))
        else:
            with open(filename,'r' ) as fp:
                line = fp.read()
                type,truncated,occluded,x1,y1,x2,y2,h,w,l,x,y,z,ry,score = line.split()
                print('{} {} x1 {} y1 {} x2 {} y2 {}'.format(filename,type,x1,y1,x2,y2))