__author__ = 'jeremy'

# NOTE - THE VERSION WHICH IS ON EXTREMELY IS NOT ACTUALLY USED
# the real pd.py is   on braini , and this one is here only so that I can enqueue the right function name
# the mightili version should be pulled from our repo using git pull , so actually the files are identical.
# but don't expect changes to pd.py on extremeli to hchange anything until you do a git pull on mightili
# (or wherever pd.py is actually running)

# paperdoll run from matlab
# this needs to run on server that
# has matlab running w. 2opencv3
# which is currently 'mightili'

import subprocess
import imghdr
from contextlib import contextmanager
import random
import string
import time
import os
import json
import sys
import logging
import  StringIO
import numpy as np
import cv2

import matlab.engine
from .. import Utils
from .. import constants

logging.basicConfig(level=logging.DEBUG)

def get_parse_from_matlab(image_filename):
    with run_matlab_engine() as eng:
        mask, label_names, pose = eng.pd("inputimg.jpg", nargout=3)
        label_dict = dict(zip(label_names, range(0, len(label_names))))
        # logging.debug('label dict'+str(label_dict))
        # stripped_name = image_filename.split('.jpg')[0]
        # outfilename = 'pd_output/' + stripped_name + '.png'
        # savedlabels = 'pd_output/' + stripped_name + '.lbls'
        # savedpose = 'pd_output/' + stripped_name + '.pos'

        # logging.debug('outfilename:' + outfilename)
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
    logging.debug("this is a totally awesome test function")
    return (6 * 7)

def get_parse_mask(img_url_or_cv2_array):
    img = Utils.get_cv2_img_array(img_url_or_cv2_array)
    if img is not None and cv2.imwrite('inputimg.jpg', img):
        if 'jpeg' != imghdr.what('inputimg.jpg'):
            return [[], [], []]
        stripped_name = rand_string()  # img_url_or_cv2_array.split('//')[1]
        modified_name = stripped_name.replace('/', '_')
    #    logging.debug('stripped name:' + stripped_name)
        logging.debug('modified name:' + modified_name)

        mask, label_dict, pose = get_parse_from_matlab(modified_name)
     #   logging.debug('labels:' + str(label_dict))
        mask_np = np.array(mask, dtype=np.uint8)
        pose_np = np.array(pose, dtype=np.uint8)
        #        if callback_pack is not None:
        #            a=callback_pack[0]
        #            logging.debug('callback function returned:'+str(a))

    else:
        if img is None:
            logging.debug('image is empty in get_parse_mask')
        else:
            logging.debug('problem writing (probably) in get_parse_mask')
        return [[], [], []]


def get_parse_from_matlab_parallel(image_filename, matlab_engine, use_parfor=False):

#    >>> import matlab.engine
#>>> eng = matlab.engine.start_matlab()
#>>>
#>>> # Do something that throws an exception...
#>>>
#>>> eng.eval('exception = MException.last;', nargout=0)
#>>> eng.eval('getReport(exception)')

    with open('pd_ml_log.log','a') as f:
        f.write('starting analysis of image: '+image_filename+'\n')
        f.close()

    with open(image_filename+'.log','a') as g:
        g.write('starting pd.py analysis of image: '+image_filename+'\n')
        g.close()


    logging.debug('get_parse_from_ml_parallel is using name:' + image_filename+' and use_parfor='+str(use_parfor))
    out = StringIO.StringIO()
    err = StringIO.StringIO()
    if use_parfor:
        mask, label_names, pose = matlab_engine.pd_parfor(image_filename, nargout=3,stdout=out,stderr=err)
    else:
        mask, label_names, pose = matlab_engine.pd(image_filename, nargout=3,stdout=out,stderr=err)
    outstring = out.getvalue()
    errstring = err.getvalue()
    logging.debug('ml output:'+str(outstring)+'\n')
    logging.debug('ml err output:'+str(errstring)+'\n')
    with open('pd_ml_log.log','a') as f:
        f.write('image: '+image_filename+'\n')
        f.write('output: '+outstring+'\n')
        if errstring is not None and errstring is not '':
            f.write('err: '+errstring+'\n')
        f.close()
    if errstring is not None and errstring is not '':
        with open('pd_ml_errlog.log','a') as f:
            f.write('image: '+image_filename+'\n')
            f.write('output: '+outstring+'\n')
            f.write('err: '+errstring+'\n')
            f.close()
#    logging.debug('ml output:'+str(out.getvalue()))
 #   logging.debug('ml stderr:'+str(err.getvalue()))

    os.remove(image_filename)
    label_dict = dict(zip(label_names, range(0, len(label_names))))
#    logging.debug('mask in getparse:'+str(mask))
#    logging.debug('label dict in getparse:'+str(label_dict))
#    logging.debug('pose in getparse:'+str(pose))
    if len(mask) == 0:
        logging.debug('paperdoll failed and get_parse_fmp is returning Nones')
        save_fail_image(image_filename)
        raise Exception('paperdoll failed on this file:',image_filename)
        return None, None, None

    with open(image_filename+'.log','a') as g:
        g.write('finished analysis of image: '+image_filename+'\n')
        g.close()


    return mask, label_dict, pose

def save_fail_image(img_filename):
    img_arr = cv2.imread(img_filename)
    logging.debug('attempting to save fail image')
    if img_arr is not None:
        fail_filename = 'fail'+img_arr
        dir = constants.pd_output_savedir
        path = os.path.join(dir,fail_filename)
        cv2.imwrite(path,img_arr)
        logging.debug('sucessful save of fail image')
        return
    logging.debug('could not read image '+str(img_filename))
    return


def get_parse_mask_parallel(matlab_engine, img_url_or_cv2_array, filename=None, use_parfor=False):
    start_time=time.time()
    img = Utils.get_cv2_img_array(img_url_or_cv2_array)
    filename = filename or rand_string()
    img_ok = image_big_enough(img)
    if img_ok and cv2.imwrite(filename + '.jpg', img):
        mask, label_dict, pose = get_parse_from_matlab_parallel(filename + '.jpg', matlab_engine, use_parfor=use_parfor)
        #logging.debug('labels:' + str(label_dict))
        mask_np = np.array(mask, dtype=np.uint8)
        pose_np = np.array(pose, dtype=np.uint8)
        finish_time=time.time()
        logging.debug('..........elapsed time in get_parse_mask_parallel:'+str(finish_time-start_time)+'.........')
        logging.debug('attempting convert and save')
        if isinstance(img_url_or_cv2_array,basestring):
            url = img_url_or_cv2_array
        else:
            url = None
        convert_and_save_results(mask_np, label_dict, pose_np, filename+'.jpg', img, url)
        return mask_np, label_dict, pose_np, filename
    else:
        if img is None:
            raise ValueError("input image is empty")
        elif not img_ok:
            raise ValueError("input image is  too small")
        else:
            raise ValueError("problem writing "+str(filename)+" in get_parse_mask_parallel")

def image_big_enough(img_array):
    if img_array is None:
        logging.debug('image is Nonel')
        return False
    width = img_array.shape[0]
    height = img_array.shape[1]
    if (width < constants.minimum_im_width or height < constants.minimum_im_height):
        logging.debug('image dimensions too small')
        return False
    else:
        return True

def convert_and_save_results(mask, label_names, pose,filename,img,url):
    fashionista_ordered_categories = constants.fashionista_categories
        #in case it changes in future - as of 2/16 this list goes a little something like this:
        #fashionista_categories = ['null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse','boots',
          #                'blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings','scarf','hat',
            #              'top','cardigan','accessories','vest','sunglasses','belt','socks','glasses','intimate',
              #            'stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges','ring',
                #          'flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch','pumps','wallet',
                  #        'bodysuit','loafers','hair','skin']
    new_mask=np.ones(mask.shape)*255  # anything left with 255 wasn't dealt with
    success = True #assume innocence until proven guilty
    logging.debug('attempting convert and save')
    for label in label_names: # need these in order
        if label in fashionista_ordered_categories:
            fashionista_index = fashionista_ordered_categories.index(label) + 0  # numbered by 1=null,56=skin
            pd_index = label_names[label]
       #     logging.debug('old index '+str(pd_index)+' for '+str(label)+': gets new index:'+str(fashionista_index))
            new_mask[mask==pd_index] = fashionista_index
        else:
            logging.debug('label '+str(label)+' not found in regular cats')
            success=False
    if 255 in new_mask:
        logging.debug('didnt fully convert mask')
        success = False
    if success:
        try:
            dir = constants.pd_output_savedir
            full_name = os.path.join(dir,filename)
#            full_name = filename
            bmp_name = full_name.strip('.jpg') + ('.bmp')
            logging.debug('writing output img to '+str(full_name))
            cv2.imwrite(full_name,img)
            logging.debug('writing output bmp to '+str(bmp_name))
            cv2.imwrite(bmp_name,new_mask)
            pose_name = full_name.strip('.jpg')+'.pose'
#            logging.debug('orig pose '+str(pose))
#            logging.debug('writing pose to '+str(pose_name))
            with open(pose_name, "w+") as outfile:
                logging.debug('succesful open, attempting to write pose')
                poselist=pose[0].tolist()
#                json.dump([1,2,3], outfile, indent=4)
                json.dump(poselist,outfile, indent=4)
            if url is not None:
                url_name = full_name.strip('.jpg')+'.url'
                logging.debug('writing url to '+str(url_name))
                with open(url_name, "w+") as outfile2:
                    logging.debug('succesful open, attempting to write:'+str(url))
                    outfile2.write(url)
            return new_mask
        except:
            logging.debug('fail in convert_and_save_results dude, bummer')
            logging.debug(str(sys.exc_info()[0]))
            return
    else:
        logging.debug('didnt fully convert mask, or unkown label in convert_and_save_results')
        success = False
        return

def show_max(parsed_img, labels):
    maxpixval = np.ma.max
    logging.debug('max pix val:' + str(maxpixval))
    maxlabelval = len(labels)
    logging.debug('max label val:' + str(maxlabelval))


def test_scp():
    subprocess.Popen("scp -i ~/first_aws.pem  img.jpg ubuntu@extremeli.trendi.guru:img777.jpg", shell=True,
                     stdout=subprocess.PIPE).stdout.read()


#def analyze_dir(path):

def run_test(img_filename):
    eng = matlab.engine.start_matlab()

    # url =  'http://aelida.com/wp-content/uploads/2012/06/love-this-style.jpg'
#    url = 'http://assets.yandycdn.com/HiRez/ES-4749-B-AMPM2012-2.jpg'
    img, labels, pose = get_parse_from_matlab(img_filename)

#    img, labels, pose = get_parse_mask(img_filename)
    show_max(img, labels)
#    logging.debug('im:' + str(img))
    logging.debug('labels:' + str(labels))
    logging.debug('pose:' + str(pose))
    # show_parse(img_array=img)


@contextmanager
def run_matlab_engine(options_string='-nodesktop'):
    eng = matlab.engine.start_matlab(options_string)
    yield eng
    eng.quit()

if __name__ == "__main__":
    im = '/home/netanel/meta/dataset/test1/product_9415_photo_3295_bbox_336_195_339_527.jpg'
    run_test(im)
