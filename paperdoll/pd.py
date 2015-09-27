__author__ = 'jeremy'

#NOTE - THIS VERSION WHICH IS ON EXTREMELY IS NOT ACTUALLY USED
# the real pd.py is   on mightili , and this one is here only so that I can enqueue the right function name

# paperdoll run from matlab
# this needs to run on server that
# has matlab running w. 2opencv3
#which is currently 'mightili.trendi.guru'

import subprocess
import numpy as np
import imghdr
import cv2

from contextlib import contextmanager
import random
import string

from trendi_guru_modules import Utils


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
        return mask_np, label_dict, pose_np


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
    print('labels:'+str(labels))
    #show_parse(img_array=img)




@contextmanager
def run_matlab_engine(options_string='-nodesktop'):
    eng = matlab.engine.start_matlab(options_string)
    yield eng
    eng.quit()

if __name__ == "__main__":
    run_test()
engien