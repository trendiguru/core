__author__ = 'jeremy'

#NOTE - THIS VERSION WHICH IS ON EXTREMELY IS NOT ACTUALLY USED
# the real pd.py is   on mightili , and this one is here only so that I can enqueue the right function name

# paperdoll run from matlab
# this needs to run on server that
# has matlab running w. opencv3
#which is currently 'mightili.trendi.guru'
# paperdoll run from matlab - this needs to run on server that
# has matlab running w. opencv3 , which is currently 'mightili.trendi.guru'

import subprocess
import shutil
import time
import numpy as np

import requests

import matlab.engine


def get_parse_from_matlab(image_filename):
    eng = matlab.engine.start_matlab('-nodesktop')
    mask, label_names, pose = eng.pd("inputimg.jpg", nargout=3)
    stripped_name = image_filename.split('.jpg')[0]
    outfilename = 'pd_output/' + stripped_name + '.png'
    print('outfilename:' + outfilename)
    subprocess.Popen("scp -i /home/jeremy/first_aws.pem  output.png ubuntu@extremeli.trendi.guru:" + outfilename,
                     shell=True, stdout=subprocess.PIPE).stdout.read()
    subprocess.Popen("cp output.png " + outfilename, shell=True, stdout=subprocess.PIPE).stdout.read()
    subprocess.Popen("cp inputimg.jpg " + outfilename, shell=True, stdout=subprocess.PIPE).stdout.read()
    #scp -i ~/first_aws.pem  img.jpg ubuntu@extremeli.trendi.guru:.
    return mask, label_names, pose


def get_parse_mask(image_url=None, image_filename=None):
    if image_filename is not None:  # copy file to 'inputimg.jpg'
        subprocess.Popen("cp " + image_filename + " inputimg.jpg", shell=True, stdout=subprocess.PIPE).stdout.read()
        time.sleep(50)  # give some time for file to write
        get_parse_from_matlab(image_filename)
        return
    response = requests.get(image_url, stream=True)
    with open('inputimg.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    time.sleep(50)  # give some time for file to write
    # img_array = imdecode(np.asarray(bytearray(response.content)), 1)
    stripped_name = image_url.split('//')[1]
    modified_name = stripped_name.replace('/', '_')
    print('stripped name:' + stripped_name)
    print('modified name:' + modified_name)
    #        cv2.imwrite(img_array,stripped_name)
    mask, label_names, pose = get_parse_from_matlab(modified_name)
    print('labels:' + str(label_names))
    label_dict = dict(zip(label_names, range(0, len(label_names))))

    return mask, label_dict, pose


def show_max(parsed_img, labels):
    maxpixval = np.ma.max
    print('max pix val:' + str(maxpixval))
    maxlabelval = len(labels)
    print('max label val:' + str(maxlabelval))


def test_scp():
    subprocess.Popen("scp -i ~/first_aws.pem  img.jpg ubuntu@extremeli.trendi.guru:img777.jpg", shell=True,
                     stdout=subprocess.PIPE).stdout.read()

# os.system("scp -i ~/first_aws.pem  img.jpg ubuntu@extremeli.trendi.guru:.")

if __name__ == "__main__":
    img, labels, pose = get_parse_mask('img.jpg')
    show_max(img, labels)
    show_parse(img_array=img)

# import matlab.engine
# eng = matlab.engine.start_matlab("nodisplay")
# print('factors of 100:')
# f = eng.factor(100)
# print(f)
