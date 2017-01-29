import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import requests

# def grabcut(img):
# 
#     img = cv2.imread(img)

def grabCut(url_or_np_array):

    print "Starting the face detector testing!"
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        # img = url_to_image(url_or_np_array)
        response = requests.get(url_or_np_array)  # download
        img = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    elif type(url_or_np_array) == np.ndarray:
        img = url_or_np_array
    else:
        return None

    rect = img.shape
    # rect = sys.argv[2]

    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)


    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    # plt.imshow(img),plt.colorbar(),plt.show()
    # cv2.imwrite("/data/yonatan/yonatan_files/grabcut_image.jpg", img)

    print cv2.imwrite("/data/yonatan/linked_to_web/grabcut_testing.jpg", img)
    # return img
