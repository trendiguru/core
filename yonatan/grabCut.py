import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import requests

# def grabcut(img):
#
#     img = cv2.imread(img)


def grabcut(url_or_np_array):

    print "Starting the face detector testing!"
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        # img = url_to_image(url_or_np_array)
        response = requests.get(url_or_np_array)  # download
        img = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    elif type(url_or_np_array) == np.ndarray:
        img = url_or_np_array
    else:
        print "couldn't open the image"
        return None

    print "image.shape: {0}\nimage.shape[0]: {1}\nimage.shape[1]: {2}".format(img.shape, img.shape[0], img.shape[1])
    rect = (5, 5, img.shape[1] - 5, img.shape[0] - 15)
    ## rect is in the form of (x, y, w, h)
    # rect = sys.argv[2]

    mask = np.zeros(img.shape[:2] ,np.uint8)

    bgdModel = np.zeros((1 ,65) ,np.float64)
    fgdModel = np.zeros((1 ,65) ,np.float64)

    cv2.grabCut(img ,mask ,rect ,bgdModel ,fgdModel ,5 ,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where(( mask ==2 ) |( mask ==0) ,0 ,1).astype('uint8')
    without_bg_img = img* mask2[:, :, np.newaxis]

    print "type(img): {0}".format(type(img))

    i, j = np.where(mask2)
    indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                          np.arange(min(j), max(j) + 1),
                          indexing='ij')
    sub_image = img[indices]


    # plt.imshow(img),plt.colorbar(),plt.show()

    # print cv2.imwrite("/data/yonatan/linked_to_web/grabcut_testing.jpg", without_bg_img)
    # print cv2.imwrite("/data/yonatan/linked_to_web/grabcut_sub_image.jpg", sub_image)

    return sub_image

