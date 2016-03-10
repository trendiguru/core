__author__ = 'yuli'


from skimage.feature import hog
from skimage import data, color, exposure
import Utils
import background_removal
import kassper
import cv2
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image



def yuli_fp(trimmed_mask, feature_size):
    from itertools import product, chain
    import MR8filters

    #sample_texture (trimmed_mask , feature_size= min_dim/10 , n= 1 ):

    centy = trimmed_mask.shape[0]/2 ; centx = trimmed_mask.shape[1]/2
    # img = cv2.rectangle(trimmed_mask ,(centx-feature_size, centy-feature_size),(centx+feature_size, centy+feature_size),(255, 0, 0))
    # cv2.imshow('showrect',img)
    # cv2.imwrite('showrect.jpg',img)
    # cv2.waitKey(0)

    trimmed_mask = trimmed_mask[ centy-feature_size/2:centy+feature_size/2, centx-feature_size/2:centx+feature_size/2]
    print trimmed_mask.shape

    #feature = (feature - np.mean(feature))/np.std(feature)

    fd, hog_image = hog(trimmed_mask, orientations=16, pixels_per_cell=(2,2),
                        cells_per_block=(1, 1), visualise=True, normalise=False)
    print fd.shape
    #print fd
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(trimmed_mask, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    return fd

def mean_fd_hog(fd, orient=16):
    # response dimentions should be mod(n=8)
    m_fd = []
    index = range(len(fd))
    m_fd.append(np.mean(fd[index[::orient]]))
    for i in (np.asarray(range(orient-1))+1):
        #c = [fd[index] for index in b]
        m_fd.append(np.mean(fd[index[i::orient]]))

    print 'm_fd shape:', np.asarray(m_fd).shape
    return np.asarray(m_fd)


if __name__ == "__main__":

 
    image = Utils.get_cv2_img_array("text3.jpg")
            #small_image, resize_ratio = background_removal.standard_resize(image, 400)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    samp_size = 40
    fd = yuli_fp(gray_img, samp_size )
    print 'fd size:', fd.shape

    #m_fds.append(mean_fd_hog(fd))
    m_fd = mean_fd_hog(fd)
    print 'len m_fd: ',len(m_fd)

        # with open(path_out+str(idx)+'_hog2X216.pickle', 'w') as f:
        #     pickle.dump(m_fds, f)










