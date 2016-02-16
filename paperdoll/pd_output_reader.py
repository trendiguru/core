import cv2
import os
import pwd
import numpy as np
#print os.environ["USER"]
#print os.getuid() # numeric uid
#print pwd.getpwuid(os.getuid())
#print os.environ["OLDPWD"]
if not 'REDIS_HOST' in os.environ:
    os.environ['REDIS_HOST'] = 'localhost'
    os.environ['REDIS_PORT'] = '6379'
if not 'MONGO_HOST' in os.environ:
    os.environ['MONGO_HOST'] = 'localhost'
    os.environ['MONGO_PORT'] = '27017'
from trendi import constants

import paperdoll_parse_enqueue

def show_pd_results(file_base_name):
    mask_file=file_base_name+'.bmp'
    jpg_file=file_base_name+'.jpg'
    pose_file=file_base_name+'.pose'

    print('reading '+str(jpg_file))
    img_arr = cv2.imread(jpg_file)
    if img_arr is not None:
        cv2.imshow('orig',img_arr)
    else:
        print('coundlt get jpg at '+str(jpg_file))

 #   paperdoll_parse_enqueue.colorbars()

    print('reading '+str(mask_file))
    mask_arr = cv2.imread(mask_file)
    if mask_arr is not None:
        paperdoll_parse_enqueue.show_parse(img_array = mask_arr)
    else:
        print('couldnt get png at '+str(mask_file))
    max = np.amax(mask_arr)
    min = np.amin(mask_arr)
    print('min {0} max {0} '.format(min,max))
    mask_arr = mask_arr-1
    maxVal = 50  # 31 categories in paperdoll
    scaled = np.multiply(mask_arr, int(255 / maxVal))
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_HOT)
    h,w,d = img_arr.shape
    print('h {0} w {1} d{2} '.format(h,w,d))
    both = np.concatenate((img_arr,colored), axis=1)
#    new_image = np.zeros([h,2*w,3])
 #   new_image[:,0:w,:] = img_arr
  #  new_image[:,w:,:] = dest


    cv2.imshow("orig", img_arr)
    cv2.imshow("dest", colored)
    cv2.imshow("both", both)
    cv2.waitKey(1000)

def colorbars(max=55):
    fashionista_ordered_categories = constants.fashionista_categories
        #in case it changes in future - as of 2/16 this list goes a little something like this:
        #fashionista_categories = ['null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse','boots',
          #                'blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings','scarf','hat',
            #              'top','cardigan','accessories','vest','sunglasses','belt','socks','glasses','intimate',
              #            'stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges','ring',
                #          'flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch','pumps','wallet',
                  #        'bodysuit','loafers','hair','skin']
    bar_height = 10
    bar_width = 20
    new_img = np.ones([max*bar_height,bar_width],np.uint8)
    for i in range(0,max):
        new_img[i*bar_height:(i+1)*bar_height,:] = int(i*255.0/max)
    #print(new_img)
    cv2.imwrite('testvarout.jpg',new_img)
    print('writing file')
 #   cv2.imshow('colorbars',new_img)
 #   cv2.waitKey(0)

    show_parse(img_array=new_img+1)




if __name__ == '__main__':
    print os.getuid() # numeric uid
    print pwd.getpwuid(os.getuid())
    path = '/home/jr/tg/pd_output'
    path = '/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain2'
    files = ['56558cd462532224c676ba7c']
    #take the file 'base' i.e. without extension
    files = [f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    print('nfiles:'+str(len(files)))

    for file in files:
        fullpath = os.path.join(path,file)
        show_pd_results(fullpath)
#        raw_input('enter for next')


