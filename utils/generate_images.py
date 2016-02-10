import cv2
import numpy as np
# import scipy as sp
import os


def generate_images(img_filename, max_angle = 5,n_angles=10,
                    max_offset_x = 100,n_offsets_x=1,
                    max_offset_y = 100, n_offsets_y=1,
                    max_scale=1.2, n_scales=1,
                    noise_level=0.05,n_noises=1,
                    blur_level=2, n_blurs=1,
                    do_mirror_lr=True,do_mirror_ud=False,output_dir='./'):
    '''
    generates a bunch of slight variations of image by rotating, translating, noising
    total # images generated is n_angles*n_offsets_x*n_offsets_y*n_noises*n_scales, these are done in nested loops
    if you don't want a particular xform set n_whatever = 1
    original image dimensions are preserved
    :param img_arr: image array to vary
    :param max_angle: rotation limit (degrees)
    :param n_angles: number of rotated images
    :param max_offset_x: x offset limit (pixels)
    :param n_offsets_x: number of x-offset images
    :param max_offset_y: y offset limit (pixels)
    :param n_offsets_y: number of y-offset images
    :param max_scales: global scaling factor
    :param n_scales: number of globally scaled images
    :param noise_level: level of gaussian noise to add - 0->no noise, 1->noise_level (avg 128)
    :param n_noises: number of noised images
    :param blur_level: level of gaussian noise to add - 0->no noise, 1->noise_level (avg 128)
    :param n_blurs: number of noised images
    :param do_mirror_lr: work on orig and x-axis-flipped copy
    :param do_mirror_ud: work on orig and x-axis-flipped copy
    :param output_dir: dir to write output images
    :return:
    '''

    img_arr = cv2.imread(img_filename)
    if img_arr is None:
        print('didnt get input image')
        return

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    eps = 0.01
    if n_angles <2:
        angles = [0.0]
    else:
        angles = np.arange(-max_angle, max_angle+eps, max_angle*2 / (n_angles-1))
    if n_offsets_x <2:
        offsets_x = [0]
    else:
        offsets_x = np.arange(-max_offset_x, max_offset_x+eps, max_offset_x*2/(n_offsets_x-1))
    if n_offsets_y <2:
        offsets_y = [0]
    else:
        offsets_y = np.arange(-max_offset_y, max_offset_y+eps, max_offset_y*2/(n_offsets_y-1))
    if n_scales <2:
        scales = [1.0]
    else:
        scales = np.arange(1, max_scale, max_scale/(n_scales))

    print('angles {0} offsets_x {1} offsets_y {2} scales {3} n_noises {4} lr {5} ud {6}'.format(angles,offsets_x,offsets_y,scales,n_noises,do_mirror_lr,do_mirror_ud))

    height=img_arr.shape[0]
    width=img_arr.shape[1]
    center = (width/2,height/2)


    reflections=[img_arr]
    if do_mirror_lr:
        fimg=img_arr.copy()
        mirror_image = cv2.flip(fimg,0)
        reflections.append(mirror_image)
    if do_mirror_ud:
        fimg=img_arr.copy()
        mirror_image = cv2.flip(fimg,1)
        reflections.append(mirror_image)
    if do_mirror_ud and do_mirror_lr:
        fimg=img_arr.copy()
        mirror_image = cv2.flip(fimg,0)
        mirror_image = cv2.flip(mirror_image,1)
        reflections.append(mirror_image)

    cv2.imshow('orig',img_arr)
    k = cv2.waitKey(0)
    #SO CLEANNNN
    for n_reflection in range(0,len(reflections)):
        for offset_x in offsets_x:
            for offset_y in offsets_y:
                for angle in angles:
                    for scale in scales:
                        for i in range(0,n_noises+1):
                            for blur in blurs:

                                print('center {0} angle {1} scale {2} h {3} w {4}'.format(center,angle, scale,height,width))
                                M = cv2.getRotationMatrix2D(center, angle,scale)
                                print('M='+str(M))
                                M[0,2]=M[0,2]+offset_x
                                M[1,2]=M[1,2]+offset_y
                                print('M='+str(M))
                                xformed_img_arr = cv2.warpAffine(reflections[n_reflection],  M, (height,width))
                                name = img_filename[0:-4]+'_ref{0}dx{1}dy{2}rot{3}scl{4}noise{5}'.format(n_reflection,offset_x,offset_y,angle,scale,i)+'.jpg'
                                print('name:'+str(name))
                                cv2.imwrite(name, xformed_img_arr)
                                cv2.imshow('xformed',xformed_img_arr)
                                k = cv2.waitKey(0)
                          #  raw_input('enter to cont')

if __name__=="__main__":
    img_filename = '../images/female1.jpg'
    generate_images(img_filename, max_angle = 10,n_angles=3, max_offset_x = 100,n_offsets_x=3,  max_offset_y = 100, n_offsets_y=3,
                    max_scale=1.5, n_scales=2, noise_level=0.05,n_noises=0,do_mirror_lr=True,do_mirror_ud=False,output_dir='./')
