__author__ = 'Nadav Paz'

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import cv2
import sklearn.mixture

from trendi import background_removal
from trendi import kassper

def test_skin_from_face(img_arr):
    ff_cascade = background_removal.find_face_cascade(img_arr, max_num_of_faces=10)
    print('ffcas:'+str(ff_cascade))
    if ff_cascade['are_faces'] :
        faces = ff_cascade['faces']
        if faces == []:
            print('ffascade reported faces but gave none')
        else:
            img_ff_cascade = img_arr.copy()

            for face in faces:
                print('cascade face:{}'.format(face))
                #make smaller rectanbgle to include less hair/bg
                margin = 0.1
                smaller_face = [int(face[0]+face[2]*margin),int(face[1]+face[3]*margin),int(face[2]*(1-2*margin)), int(face[3]*(1-2*margin))]
                face = smaller_face
                cv2.rectangle(img_ff_cascade,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,100,0),2)
#                fsc = face_skin_color_estimation_gmm(img_arr,face,visual_output=True)

                fsc = background_removal.face_skin_color_estimation_gmm(img_arr,face,visual_output=True)
                print('face skin color {}'.format(fsc))
                f=0.5
                ranges = [[90,240], #,int(fsc[0][0]-(fsc[0][1]/f)),  #change means, stdvs to ranges.  force y chan to known range
                         # int(fsc[0][0]+(fsc[0][1]/f))],
                          [int(fsc[1][0]-(fsc[1][1]/f)),
                          int(fsc[1][0]+(fsc[1][1]/f))],
                          [int(fsc[2][0]-(fsc[2][1]/f)),
                          int(fsc[2][0]+(fsc[2][1]/f))]]
                print('ranges:'+str(ranges))

                mask = kassper.skin_detection_fast(img_arr,ycrcb_ranges = ranges)
                cv2.imshow('skin mask',mask*255)
                mask2 = kassper.skin_detection_fast(img_arr)
                cv2.imshow('skin mask_norange',mask2*255)
            cv2.imshow('ffcascade',img_ff_cascade)
            cv2.waitKey(0)


# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))



def face_skin_color_estimation(image, face_rect):
    x, y, w, h = face_rect
    face_image = image[y:y + h, x:x + w, :]
    face_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    n_pixels = face_image.shape[0]*face_image.shape[1]
    print('npixels:'+str(n_pixels))
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    # Define some test data which is close to Gaussian
    channels = [np.ravel(face_hsv[:,:,0]),np.ravel(face_hsv[:,:,1]),np.ravel(face_hsv[:,:,2])]
    labels = ['h','s','v']
    for data,label in zip(channels,labels):
        hist, bin_edges = np.histogram(data, density=False)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        p0 = [1., 0., 1.]
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
        # Get the fitted curve
        hist_fit = gauss(bin_centres, *coeff)
        plt.plot(bin_centres, hist,'.-', label='Test data '+label)
        plt.plot(bin_centres, hist_fit, 'o-',label='Fitted data '+label)
        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        print 'Fitted mean = ', coeff[1]
        print 'Fitted standard deviation = ', coeff[2]
        print('coeff 0 '+str(coeff[0]))
    plt.legend()
    plt.show()

def face_skin_color_estimation_gmm(image, face_rect,visual_output=False):
    '''
    get params of skin color - gaussian approx for h,s,v (independently)
    :param image:
    :param face_rect:
    :param visual_output:
    :return: [(h_mean,h_std),(s_mean,s_std),(v_mean,v_std)]
    '''

    x, y, w, h = face_rect
    face_image = image[y:y + h, x:x + w, :]
    face_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    n_pixels = face_image.shape[0]*face_image.shape[1]
    print('npixels:'+str(n_pixels))
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    # Define some test data which is close to Gaussian
    gmm = sklearn.mixture.GMM()
#    r = gmm.fit(face_hsv) # GMM requires 2D data as of sklearn version 0.16
    channels = [np.ravel(face_hsv[:,:,0]),np.ravel(face_hsv[:,:,1]),np.ravel(face_hsv[:,:,2])]
    labels = ['h','s','v']
    results = []
    for data,label in zip(channels,labels):
        r = gmm.fit(data[:,np.newaxis]) # GMM requires 2D data as of sklearn version 0.16
        print("mean : %f, var : %f" % (r.means_[0, 0], r.covars_[0, 0]))
        results.append((r.means_[0, 0], r.covars_[0, 0]))
        hist, bin_edges = np.histogram(data, density=False)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        p0 = [1., 0., 1.]
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
        # Get the fitted curve

        if visual_output:
            plt.plot(bin_centres, hist,'.-', label='Test data '+label)
        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
    if visual_output:
        plt.legend()
        plt.show()
    return results


def compare_detection_methods(img_arr):
    cv2.imshow('orig',img_arr)

    ffc = background_removal.find_face_ccv(img_arr, max_num_of_faces=100)
    print('ccv:'+str(ffc))
    if ffc['are_faces'] :
        faces = ffc['faces']
        img_ffc = img_arr.copy()
        for face in faces:
            print('ccv face:{}'.format(face))
            cv2.rectangle(img_ffc,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,100,0),5)
        cv2.imshow('ffc',img_ffc)

    ff_cascade = background_removal.find_face_cascade(img_arr, max_num_of_faces=10)
    print('ffcas:'+str(ff_cascade))
    if ff_cascade['are_faces'] :
        faces = ff_cascade['faces']
        if faces == []:
            print('ffascade reported faces but gave none')
        else:
            img_ff_cascade = img_arr.copy()

            for face in faces:
                print('cascade face:{}'.format(face))
                cv2.rectangle(img_ff_cascade,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,100,0),5)
            cv2.imshow('ffcascade',img_ff_cascade)


    ff_dlib = background_removal.find_face_dlib(img_arr, max_num_of_faces=100)
    print('ffdlib:'+str(ff_dlib))
    if ff_dlib['are_faces'] :
        faces = ff_dlib['faces']
        img_ff_dlib = img_arr.copy()
        for face in faces:
            print('dlib face:{}'.format(face))
            cv2.rectangle(img_ff_dlib,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,100,0),5)
        cv2.imshow('ff_dlib',img_ff_dlib)
    cv2.waitKey(0)

def check_dir(dir):
    print('doing dir {}'.format(dir))
    files = [os.path.join(dir,f) for f in os.listdir(dir) if not os.path.isdir(os.path.join(dir,f))]
    print('{} files in {}'.format(len(files),dir))
    for f in files :
        img_arr = cv2.imread(f)
        if img_arr is None:
            print('got none img arr for {}'.format(f))
            continue
#        compare_detection_methods(img_arr)
        test_skin_from_face(img_arr)

if __name__ == "__main__":
    dir = '/media/jeremy/9FBD-1B00/data/jeremy/image_dbs/mongo/amazon_us_female/suit/'
    dir = '/data/jeremy/image_dbs/mongo/amazon_us_female/suit/'
    check_dir(dir)
    # subdirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    # print('{} subdirs in {}'.format(len(subdirs),dir))
    # for subdir in subdirs:
    #     check_dir(subdir)



    # background_removal.choose_faces(img_arr, faces_list, max_num_of_faces)
    # background_removal.score_face(face, image)
    # background_removal.face_is_relevant(image, face)
    # background_removal.is_skin_color(face_ycrcb):
# variance_of_laplacian(image):
# is_one_color_image(image):
# average_bbs(bb1, bb2):
# combine_overlapping_rectangles(bb_list):
# body_estimation(image, face):
# bb_mask(image, bounding_box):
# paperdoll_item_mask(item_mask, bb):
# create_mask_for_gc(rectangles, image):
# create_arbitrary(image):
#
# face_skin_color_estimation(image, face_rect):
# person_isolation(image, face):
# check_skin_percentage(image):
