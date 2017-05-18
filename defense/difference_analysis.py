__author__ = 'jeremy'
'''
the point of this is  to try to take advantage of the fact that images are coming from more-or-less fixed camera
1. by eye select relatively uncluttered bgnd pics
2. take diffs bet. incoming cam images and each of those
3. find stats of diff (e.g histogram of sum over abs vals of diff. images)
4. if stats work out well (e.g. most within small range of 0 and some outliers indicating diferent cams)
then try using diffs as input (to training as well)
'''
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def analyze_difference_images(img_dir='/home/jeremy/yolosaves',bgnd_images_dir='/home/jeremy/hls_bgnds'):
    incoming_images = [os.path.join(img_dir,f) for f in os.listdir(img_dir)]
    bgnd_images = [os.path.join(bgnd_images_dir,f) for f in os.listdir(bgnd_images_dir)]
    print('n incoming {} n bgnd {}'.format(len(incoming_images),len(bgnd_images)))

    diffs = []
    count = 0
    maxcount = 300
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter('/home/jeremy/diffs.avi',fourcc,fps=28,frameSize=(720,576))
#    out = cv2.VideoWriter('/home/jeremy/diffs.avi',fourcc,fps=28,frameSize=(360,576/2))

    for img in incoming_images:
        for bgnd in bgnd_images:
            incoming_arr = cv2.imread(img)
            bgnd_arr = cv2.imread(bgnd)
            if incoming_arr is None or bgnd_arr is None:
                continue
            print(incoming_arr.shape)
            # if incoming_arr.shape[0]!=704 or incoming_arr.shape[1]!=480:
            #     continue
            if incoming_arr.shape != bgnd_arr.shape:
                print('different shapes incomig {} bgnd {}'.format(incoming_arr.shape,bgnd_arr.shape))
                continue
            diff = abs(incoming_arr - bgnd_arr)
            cv2.imshow('diff',diff)
            incoming_registered = image_register(bgnd_arr,incoming_arr)
            if incoming_registered is not None:
                incoming_arr = incoming_registered
                print('using registered')
            s = np.sum(diff)/(incoming_arr.shape[0]*incoming_arr.shape[1]*incoming_arr.shape[2])
            print('sum of diff image:'+str(s)+' shape '+str(incoming_arr.shape))
            diffs.append(s)
            count = count+1
#            diff=cv2.resize(diff,(360,576/2))
#            out.write(diff)
            if s>140:
                print('likely different camera')
            elif s<100:
                print('likely same camera')
            else:
                print('unclear difference range')
            cv2.imshow('in',incoming_arr)
            cv2.imshow('back',bgnd_arr)
            cv2.waitKey(0)
        if count>maxcount:
            break
    diffs=np.array([diffs])
    diffs=np.transpose(diffs)
    plt.hist(diffs,bins=1000)
    img_arr=cv2.imread('/home/jeremy/Dropbox/tg/diffs.png')
    img_arr=cv2.resize(img_arr,(360,576/2))

    plt.show()
 #   out.write(img_arr)
  #  out.release()

def image_register(im1,im2):
    #se e  http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/, there is ,more on gradient
    # Read the images to be aligned
   # im1 =  cv2.imread("images/image1.jpg");
   # im2 =  cv2.imread("images/image2.jpg");

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    except:
        print('had trouble registering')
        return None
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    # Show final results
#    cv2.imshow("Image 1", im1)
#    cv2.imshow("Image 2", im2)
#    cv2.imshow("Aligned Image 2", im2_aligned)
#    cv2.waitKey(0)
    return im2_aligned

if __name__ == "__main__":
    analyze_difference_images()