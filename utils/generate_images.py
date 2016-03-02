import cv2
import numpy as np
# import scipy as sp
import os


def generate_images(img_filename, max_angle = 5,n_angles=10,
                    max_offset_x = 100,n_offsets_x=1,
                    max_offset_y = 100, n_offsets_y=1,
                    max_scale=1.2, n_scales=1,
                    noise_level=0.05,n_noises=1,noise_type='gauss',
                    max_blur=2, n_blurs=1,
                    do_mirror_lr=True,do_mirror_ud=False,output_dir=None):
    '''
    generates a bunch of variations of image by rotating, translating, noising etc
    total # images generated is n_angles*n_offsets_x*n_offsets_y*n_noises*n_scales*etc, these are done in nested loops
    if you don't want a particular xform set n_whatever = 0
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
    :param noise_type     'gauss'     Gaussian-distributed additive noise.
                                            'poisson'   Poisson-distributed noise generated from the data.
                                            's&p'       Replaces random pixels with 0 or 1.
                                            'speckle'   Multiplicative noise using out = image + n*image
                                            None
    :param max_blur: level of blur (pixels in kernel) to add - 0->no noise,
    :param n_blurs: number of blurred images
    :param do_mirror_lr: work on orig and x-axis-flipped copy
    :param do_mirror_ud: work on orig and x-axis-flipped copy
    :param output_dir: dir to write output images
    :return:
    '''

    img_arr = cv2.imread(img_filename)
    if img_arr is None:
        print('didnt get input image')
        return
    orig_path, filename = os.path.split(img_filename)
    if output_dir is not None and not os.path.exists(output_dir):
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
    if n_scales <1:
        scales = [1.0]
    elif n_scales ==1:  #todo - change dx , dy , angles to have ==1 case
        scales = [max_scale]
    else:
        scales = np.arange(1, max_scale+eps, (max_scale-1)/(n_scales-1))
    if n_blurs <1:
        blurs = [0]
    elif n_blurs ==1:
        blurs = [max_blur]
    else:
         blurs = np.arange(1, max_blur+eps, (max_blur-1)/(n_blurs-1))
    if n_noises <1:
         n_noises=1
         noise_type=None
    print('angles {0} offsets_x {1} offsets_y {2} scales {3} n_noises {4} lr {5} ud {6} blurs {7} '.format(angles,offsets_x,offsets_y,scales,n_noises,do_mirror_lr,do_mirror_ud,blurs))

    height=img_arr.shape[0]
    width=img_arr.shape[1]
    center = (width/2,height/2)
    reflections=[img_arr]
    if do_mirror_lr:
        fimg=img_arr.copy()
        mirror_image = cv2.flip(fimg,1)
        reflections.append(mirror_image)
    if do_mirror_ud:
        fimg=img_arr.copy()
        mirror_image = cv2.flip(fimg,0)
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
                        for i in range(0,n_noises):
                            for blur in blurs:
                                original_img = reflections[n_reflection]
                                if blur == 0:
                                    blurred = original_img  #blur=0 crashes cv2.blur
                                else:
                                    blurred = cv2.blur(original_img,(int(blur),int(blur)))   #fails if blur is nonint or 0
                                noised = add_noise(blurred,noise_type,noise_level)
                                print('center {0} angle {1} scale {2} h {3} w {4}'.format(center,angle, scale,height,width))
                                M = cv2.getRotationMatrix2D(center, angle,scale)
#                                print('M='+str(M))
                                M[0,2]=M[0,2]+offset_x
                                M[1,2]=M[1,2]+offset_y
                                print('M='+str(M))
                                dest =
                                xformed_img_arr = cv2.warpAffine(noised,  M, dest, (width,height))
                                name = filename[0:-4]+'_ref{0}dx{1}dy{2}rot{3}scl{4}n{5}b{6}'.format(n_reflection,offset_x,offset_y,angle,scale,i,blur)+'.jpg'
                                if output_dir is not None:
                                    full_name = os.path.join(output_dir,name)
                                else:
                                    full_name = os.path.join(orig_path,name)
                                print('name:'+str(full_name))
                                cv2.imwrite(full_name, xformed_img_arr)
                                cv2.imshow('xformed',xformed_img_arr)
                                k = cv2.waitKey(0)
                          #  raw_input('enter to cont')


def add_noise(image, noise_typ,level):
    '''
    Parameters
    ----------
    image : ndarray
    Input image data. Will be converted to float.
    mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    '''
    print('adding noise type {0} level {1}'.format(noise_typ,level))
    if noise_typ == None:
        return image
    if noise_typ == "gauss":
        row,col,ch= image.shape
        print('row {} col {} ch {}'.format(row,col,ch))
        mean = 0
        var = level*255
        sigma = var**0.5
        print('sigma {0} mean {1}'.format(sigma,mean))
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        #        z=np.multiply(gauss,0)
 #       gauss = np.maximum(gauss,z)
        gauss = gauss.reshape(row,col,ch)
        #       cv2.imshow('orig',gauss)
  #     k = cv2.waitKey(0)
        noisy = (image + gauss)
        noisy =  noisy.astype(dtype=np.uint8)

        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = level
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = [255,255,255]
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = [0,0,0]
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy


if __name__=="__main__":
    img_filename = '../images/female1.jpg'
    generate_images(img_filename, max_angle = 3,n_angles=2,
                    max_offset_x = 50,n_offsets_x=2,
                    max_offset_y = 50, n_offsets_y=2,
                    max_scale=1.2, n_scales=2,
                    noise_level=0.1,noise_type='gauss',n_noises=2,
                    max_blur=5, n_blurs=2,
                    do_mirror_lr=True,do_mirror_ud=False,output_dir='snorb')