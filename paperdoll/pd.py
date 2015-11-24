__author__ = 'jeremy'

# NOTE - THE VERSION WHICH IS ON EXTREMELY IS NOT ACTUALLY USED
# the real pd.py is   on mightili , and this one is here only so that I can enqueue the right function name
# the mightili version should be pulled from our repo using git pull , so actually the files are identical.
# but don't expect changes to pd.py on extremeli to hchange anything until you do a git pull on mightili
# (or wherever pd.py is actually running)

# paperdoll run from matlab
# this needs to run on server that
# has matlab running w. 2opencv3
# which is currently 'mightili.trendi.guru'

import subprocess
import imghdr
from contextlib import contextmanager
import random
import string
import time
import os
import pickle
import numpy as np
import cv2

import matlab.engine
from .. import Utils
from .. import constants

def get_parse_from_matlab(image_filename):
    with run_matlab_engine() as eng:
        mask, label_names, pose = eng.pd("inputimg.jpg", nargout=3)
        label_dict = dict(zip(label_names, range(0, len(label_names))))
        # print('label dict'+str(label_dict))
        # stripped_name = image_filename.split('.jpg')[0]
        # outfilename = 'pd_output/' + stripped_name + '.png'
        # savedlabels = 'pd_output/' + stripped_name + '.lbls'
        # savedpose = 'pd_output/' + stripped_name + '.pos'

        # print('outfilename:' + outfilename)
        # subprocess.Popen("scp -i /home/jeremy/first_aws.pem  output.png ubuntu@extremeli.trendi.guru:" + outfilename,
        #                  shell=True, stdout=subprocess.PIPE).stdout.read()
        # subprocess.Popen("cp output.png " + outfilename, shell=True, stdout=subprocess.PIPE).stdout.read()
        # subprocess.Popen("cp inputimg.jpg " + outfilename, shell=True, stdout=subprocess.PIPE).stdout.read()
        # subprocess.Popen("cp savedlabels.p " + savedlabels, shell=True, stdout=subprocess.PIPE).stdout.read()
        # subprocess.Popen("cp savedpose.p " + savedpose, shell=True, stdout=subprocess.PIPE).stdout.read()

        return mask, label_dict, pose


def rand_string():
    return ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(32)])


def test_function():
    print("this is a totally awesome test function")
    return (6 * 7)


def get_parse_mask(img_url_or_cv2_array):
    img = Utils.get_cv2_img_array(img_url_or_cv2_array)
    if img is not None and cv2.imwrite('inputimg.jpg', img):
        if 'jpeg' != imghdr.what('inputimg.jpg'):
            return [[], [], []]
        stripped_name = rand_string()  # img_url_or_cv2_array.split('//')[1]
        modified_name = stripped_name.replace('/', '_')
        print('stripped name:' + stripped_name)
        print('modified name:' + modified_name)

        mask, label_dict, pose = get_parse_from_matlab(modified_name)
        print('labels:' + str(label_dict))
        mask_np = np.array(mask, dtype=np.uint8)
        pose_np = np.array(pose, dtype=np.uint8)
        #        if callback_pack is not None:
        #            a=callback_pack[0]
        #            print('callback function returned:'+str(a))
        return mask_np, label_dict, pose_np
    else:
        print('either image is empty or problem writing')
        return [[], [], []]


def get_parse_from_matlab_parallel(image_filename, matlab_engine, use_parfor=False):
    print('get_parse_from_ml_parallel is using name:' + image_filename+' and use_parfor='+str(use_parfor))
    if use_parfor:
        mask, label_names, pose = matlab_engine.pd_parfor(image_filename, nargout=3)
    else:
        mask, label_names, pose = matlab_engine.pd(image_filename, nargout=3)
    os.remove(image_filename)
    label_dict = dict(zip(label_names, range(0, len(label_names))))
    return mask, label_dict, pose

def get_parse_mask_parallel(matlab_engine, img_url_or_cv2_array, filename=None, use_parfor=False):
    start_time=time.time()
    img = Utils.get_cv2_img_array(img_url_or_cv2_array)
    filename = filename or rand_string()
    if img is not None and cv2.imwrite(filename + '.jpg', img):
        mask, label_dict, pose = get_parse_from_matlab_parallel(filename + '.jpg', matlab_engine, use_parfor=use_parfor)
        print('labels:' + str(label_dict))
        mask_np = np.array(mask, dtype=np.uint8)
        pose_np = np.array(pose, dtype=np.uint8)
        finish_time=time.time()
        print('elapsed time in get_parse_mask_parallel:'+str(finish_time-start_time))
        print('attempting convert and save')
        convert_and_save_results(mask_np,label_dict,pose_np,filename,img)
        return mask_np, label_dict, pose_np, filename
    else:
        raise ValueError("either image is empty or problem writing")

def convert_and_save_results(mask, label_names, pose,filename,img):
    fashionista_ordered_categories = constants.fashionista_categories
    new_mask=np.ones(mask.shape)*255  # anything left with 255 wasn't dealt with
    success = True #assume innocence until proven guilty
    print('attempting convert and save')
    for label in label_names: # need these in order
        if label in fashionista_ordered_categories:
            fashionista_index = fashionista_ordered_categories.index[label]
            print('old index:'+str(label_names[label])+' new index:'+str(fashionista_index))
            new_mask[mask==i] = fashionista_index
        else:
            print('label '+str(label)+' not found in regular cats')
            success=False
    if 255 in new_mask:
        print('didnt fully convert mask')
        success = False
    if success:
        dir = constants.pd_output_savedir
        full_name = os.path.join(dir,filename)
        bmp_name = full_name.strip('.jpg') + ('.bmp')
        print('writing output bmp to '+str(bmp_name))
        cv2.imwrite(bmp_name,new_mask)
        cv2.imwrite(full_name,img)
        pose_name = full_name.strip('.jpg')+'.pose'
        print('orig pose '+str(pose))
        afile = open(pose_name, 'wb')
        pickle.dump(pose, afile)
        afile.close()

        #reload object from file
        file2 = open(pose_name, 'rb')
        read_pose = pickle.load(file2)
        file2.close()
        print('read pose '+str(read_pose))

def show_max(parsed_img, labels):
    maxpixval = np.ma.max
    print('max pix val:' + str(maxpixval))
    maxlabelval = len(labels)
    print('max label val:' + str(maxlabelval))


def test_scp():
    subprocess.Popen("scp -i ~/first_aws.pem  img.jpg ubuntu@extremeli.trendi.guru:img777.jpg", shell=True,
                     stdout=subprocess.PIPE).stdout.read()


def run_test():
    # url =  'http://aelida.com/wp-content/uploads/2012/06/love-this-style.jpg'
    url = 'http://assets.yandycdn.com/HiRez/ES-4749-B-AMPM2012-2.jpg'
    img, labels, pose = get_parse_mask(img_url_or_cv2_array=url)
    show_max(img, labels)
    print('labels:' + str(labels))
    # show_parse(img_array=img)


@contextmanager
def run_matlab_engine(options_string='-nodesktop'):
    eng = matlab.engine.start_matlab(options_string)
    yield eng
    eng.quit()


if __name__ == "__main__":
    run_test()
