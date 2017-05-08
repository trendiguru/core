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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter('/home/jeremy/diffs.avi',fourcc,fps=28,frameSize=(720,576))
    out = cv2.VideoWriter('/home/jeremy/diffs.avi',fourcc,fps=28,frameSize=(360,576/2))

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
            cv2.waitKey(1)
            s = np.sum(diff)/(incoming_arr.shape[0]*incoming_arr.shape[1]*incoming_arr.shape[2])
            print('sum of diff image:'+str(s))
            diffs.append(s)
            count = count+1
            diff=cv2.resize(diff,(360,576/2))
            out.write(diff)
        if count>maxcount:
            break
    diffs=np.array([diffs])
    diffs=np.transpose(diffs)
    plt.hist(diffs,bins=1000)
    img_arr=cv2.imread('/home/jeremy/Dropbox/tg/diffs.png')
    img_arr=cv2.resize(img_arr,(360,576/2))

    out.write(img_arr)
    plt.show()
    out.release()

if __name__ == "__main__":
    analyze_difference_images()