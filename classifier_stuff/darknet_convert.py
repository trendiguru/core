__author__ = 'jeremy'
import os
import logging
import cv2

logging.basicConfig(level=logging.DEBUG)

#as described here http://guanghan.info/blog/en/my-works/train-yolo/
#we want file w lines like:
#  box1_x1_ratio box1_y1_ratio box1_width_ratio box1_height_ratio
#these aparently correspond to filename lines in the train.txt file
# so we'll step thru dirs and append the train.txt file as well as the bbox file

def dir_to_darknet(dir, trainfile,bbfile,category_number):
    only_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    n = 0
    with open(trainfile,'a') as fp_t:
        with open(bbfile,'a') as fp_bb:
            for a_file in only_files:
                full_filename = os.path.join(dir,a_file)
                dark_bb = get_darkbb(full_filename)
#                line = str(category_number)+join(str(a for a in dark_bb))
                line = str(category_number)+' '+dark_bb[0][0:5]+' '+dark_bb[1][0:5]+' '+dark_bb[2][0:5]+' '+dark_bb[3][0:5]
                print('line to write:'+line)
                fp_bb.write(line)
                fp_t.write(full_filename)
                n = n + 1
                raw_input('enter for next')
        fp_bb.close()
    fp_t.close()
    return n


def get_darkbb(filename):
    base_filename = os.path.basename(filename)
    print('name:'+str(base_filename))
    if 'bbox_' in base_filename:
        strs = base_filename.split('bbox_')
        bb_str = strs[1]
        coords = bb_str.split('_')
        bb_x = int(coords[0])
        bb_y = int(coords[1])
        bb_w = int(coords[2])
        bb_h = coords[3].split('.')[0]  #this has .jpg or .bmp at the end
        bb_h = int(bb_h)
        bb=[bb_x,bb_y,bb_w,bb_h]
        print('bb:'+str(bb))
        if bb_h == 0:
            logging.warning('bad height encountered in imutils.resize_and_crop_image for '+str(filename))
            return None
        if bb_w == 0:
            logging.warning('bad width encountered in imutils.resize_and_crop_image for '+str(filename))
            return None
        img_arr = cv2.imread(filename)
        if img_arr is None:
            logging.warning('could not read {} , it reads as None'.format(filename))
            return None
        h,w = img_arr.shape[0:2]
        converted_bb = convert([w,h],bb)
        print('converted bb:{} im_w {} im_h {}'.format(converted_bb,w,h))
        w_conv = converted_bb[2]*w
        h_conv = converted_bb[3]*h
        x_conv = converted_bb[0]*w - w_conv/2
        y_conv = converted_bb[1]*h - h_conv/2
        print('converted bb xlates to: x:{} y:{} w:{} h:{}'.format(x_conv,y_conv,w_conv,h_conv))
        return(converted_bb)
    else:
        logging.warning('no bbox_ in filename, dont know how to get bb')


def convert(imsize, box):
    '''
    convert box [x y w h ] to box [x_center% ycenter% w% h%] where % is % of picture w or h
    :param size: [image_w,image_h]
    :param box: [x,y,w,h] of bounding box
    :return: [x_center% y_center% w% h%]
    '''
    dw = 1./imsize[0]
    dh = 1./imsize[1]
    x_middle = box[0] + box[2]/2.0
    y_middle = box[1] + box[3]/2.0
    w = box[2]
    h = box[3]
    logging.debug('x_mid {} y_mid {} w {} h {}'.format(x_middle,y_middle,w,h))
    x = x_middle*dw
    w = w*dw
    y = y_middle*dh
    h = h*dh
    return (x,y,w,h)


if __name__ == '__main__':
    dir = '/home/jeremy/tg/berg_test/cropped/test_pairs_belts'
    trainfile = '/home/jeremy/tg/trainjr.txt'
    bbfile = '/home/jeremy/tg/bbjr.txt'
    n_files = dir_to_darknet(dir,trainfile,bbfile,37)