import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

def grabcut(img, rect):

    img = cv2.imread(img)

    # rect = (50,50,450,290)
    # rect = sys.argv[2]

    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)


    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    # plt.imshow(img),plt.colorbar(),plt.show()
    cv2.imwrite("/data/yonatan/yonatan_files/grabcut_image.jpg", img)