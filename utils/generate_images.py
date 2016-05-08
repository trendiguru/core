import cv2
import numpy as np
# import scipy as sp
import os
import logging
import cv2
import numpy as np
import os
from multiprocessing import Pool
import multiprocessing
import itertools

from trendi.utils import imutils
from trendi.constants import fashionista_categories_augmented,fashionista_categories_augmented_zero_based,ultimate_21
logging.basicConfig(level=logging.DEBUG)

def generate_images(img_filename, max_angle = 5,n_angles=10,
                    max_offset_x = 100,n_offsets_x=1,
                    max_offset_y = 100, n_offsets_y=1,
                    max_scale=1.2, n_scales=1,
                    noise_level=0.05,n_noises=1,noise_type='gauss',
                    max_blur=2, n_blurs=1,
                    do_mirror_lr=True,do_mirror_ud=False,output_dir=None,
                    show_visual_output=False,bb=None,do_bb=False,suffix='.jpg'):
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
        logging.warning('didnt get input image '+str(img_filename))
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
        print('n_blurs-1:' + str(n_blurs-1))
        rat = float(max_blur)/(n_blurs-1)
        print('rat:'+str(rat))
        blurs = np.arange(1, max_blur+eps, rat)
    if n_noises <1:
         n_noises=1
         noise_type=None
    print('angles {0} offsets_x {1} offsets_y {2} scales {3} n_noises {4} lr {5} ud {6} blurs {7} '.format(angles,offsets_x,offsets_y,scales,n_noises,do_mirror_lr,do_mirror_ud,blurs))

    height=img_arr.shape[0]
    width=img_arr.shape[1]
    if len(img_arr.shape) == 2:
        depth = img_arr.shape[2]
    else:
        depth = 1
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
    if show_visual_output:
        cv2.imshow('orig',img_arr)
        k = cv2.waitKey(0)
    if 'bbox_' in img_filename and bb is None and do_bb:
        strs = img_filename.split('bbox_')
        bb_str = strs[1]
        coords = bb_str.split('_')
        bb_x = int(coords[0])
        bb_y = int(coords[1])
        bb_w = int(coords[2])
        bb_h = coords[3].split('.')[0]  #this has .jpg or .bmp at the end
        bb_h = int(bb_h)
        bb=[bb_x,bb_y,bb_w,bb_h]
        bb_points  = [[bb_x,bb_y],[bb_x+bb_w,bb_y],[bb_x,bb_y+bb_h],[bb_x+bb_w,bb_y+bb_h]]  #topleft topright bottomleft bottomright
        print('bb:'+str(bb))
        if bb_h == 0:
            logging.warning('bad height encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
            return None
        if bb_w == 0:
            logging.warning('bad width encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
            return None

# Python: cv2.transform(src, m[, dst]) -> dst
#http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#void%20transform%28InputArray%20src,%20OutputArray%20dst,%20InputArray%20m%29


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
                                dest = np.ones_like(img_arr) * 255
#                                xformed_img_arr  = cv2.warpAffine(noised,  M, (width,height),dst=dest,borderMode=cv2.BORDER_TRANSPARENT)
                                xformed_img_arr  = cv2.warpAffine(noised,  M, (width,height),dst=dest,borderMode=cv2.BORDER_REPLICATE)
                                xformed_img_arr = dest
                                if do_bb:
                                    xformed_bb_points  = np.dot(bb_points,M)
                                name = filename[0:-4]+'_ref{0}dx{1}dy{2}rot{3}scl{4}n{5}b{6}'.format(n_reflection,offset_x,offset_y,angle,scale,i,blur)+suffix
                                name = filename[0:-4]+'_m%dx%dy%dr%.2fs%.2fn%db%.2f' % (n_reflection,offset_x,offset_y,angle,scale,i,blur)+suffix
                                if output_dir is not None:
                                    full_name = os.path.join(output_dir,name)
                                else:
                                    full_name = os.path.join(orig_path,name)
                                print('name:'+str(full_name))
                                cv2.imwrite(full_name, xformed_img_arr)
                                if show_visual_output:

                                  #  img_copy = xformed_img_arr.copy()
                                    #cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None
#                                    cv2.rectangle(img_copy,pt1,pt2)

                          #          cv2.line(img_copy,pt1,pt2)
                            #        cv2.line(img_copy,pt2,pt3)
                              #      cv2.line(img_copy,pt3,pt4)
                                #    cv2.line(img_copy,pt4,pt1)
                                  #  k = cv2.waitKey(0)
                                    cv2.imshow('xformed',xformed_img_arr)
                                    k = cv2.waitKey(0)
                          #  raw_input('enter to cont')

def generate_images_for_directory(fulldir,**args):
    only_files = [f for f in os.listdir(fulldir) if os.path.isfile(os.path.join(fulldir, f))]
    for a_file in only_files:
        full_filename = os.path.join(fulldir,a_file)
        generate_images(full_filename,**args)

def generate_masks(img_filename, **kwargs):

    img_arr = cv2.imread(img_filename,cv2.IMREAD_GRAYSCALE)
    if img_arr is None:
        logging.warning('didnt get input image '+str(img_filename))
        return
    print('shape:'+str(img_arr.shape))
    if len(img_arr.shape) == 3:
        logging.warning('got 3 channel image '+str(img_filename)+', using first chan')
        img_arr = img_arr[:,:,0]
    if img_arr is None:
        logging.warning('didnt get input image '+str(img_filename))
        return
    h,w = img_arr.shape[0:2]
    uniques = np.unique(img_arr)
    n_uniques=len(uniques)
    binary_masks = np.zeros([h,w,n_uniques])
    for i in range(0,n_uniques):
        binary_masks[:,:,i] = img_arr[:,:]==uniques[i]
        cv2.imshow('mask'+str(i),binary_masks[:,:,i])
        transformed_mask = transform_image(binary_masks[:,:,i],kwargs)

    cv2.waitKey(0)

def generate_images_for_directory_of_directories(dir_of_dirs,filter= None,**args):
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))  ]
    logging.debug(str(only_dirs))
    if filter:
        only_dirs = [dir for dir in only_dirs if filter in dir  ]
    logging.debug(str(only_dirs))
    for a_dir in only_dirs:
        full_dir = os.path.join(dir_of_dirs,a_dir)
        generate_images_for_directory(full_dir,**args)

def clear_underutilized_bins(img_arr):
    h = np.histogram(img_arr,bins=57)
    print h

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

def binary_masks_from_indexed_mask(indexed_mask, n_binaries=None):
    '''
    Take mask indexed by category, and turn into binary masks . nth mask is for index n
    :param indexed_mask:
    :return: n binary masks
    '''
    if n_binaries == None:
        n_binaries = np.max(indexed_mask)
    binary_masks = np.zeros((indexed_mask.shape[0], indexed_mask.shape[1], n_binaries), dtype='uint8')
    concatenated=np.zeros([indexed_mask.shape[0],1])
    for i in range(n_binaries):
        binary_masks[:, :, i][indexed_mask == i] = 1

#        cv2.imshow('mask '+str(i),(binary_masks[:,:,i]*255).astype('uint8'))
        concatenated=np.concatenate((concatenated,binary_masks[:,:,i]),1)
    conc_h,conc_w = concatenated.shape
    if conc_w > 2000:
        factor = 2000.0/conc_w
        concatenated = cv2.resize(concatenated,(int(round(factor*conc_w)),int(round(factor*conc_h))))
#    cv2.imshow('conc',concatenated)
#    cv2.imshow('indexed',indexed_mask)
#    cv2.waitKey(0)
    binary_masks = binary_masks.astype('uint8')
    return binary_masks

def indexed_mask_from_binary_masks(binary_masks):
    '''
    Take mask indexed by category, and turn into binary masks . nth mask is for index n
    :param indexed_mask:
    :return: n binary masks
    '''
    print('maskshape:'+str(binary_masks.shape))
    h,w,d = binary_masks.shape[0:3]
    indexed_mask = np.zeros((h, w), dtype='uint8')
    for i in range(d):
        indexed_mask[:, :][binary_masks[:,:,i] == 1] = i

    print('indexedshape:'+str(indexed_mask.shape))
    return indexed_mask

def show_binary_mask(binary_masks, n_binaries=None):
    '''
    Take mask indexed by category, and turn into binary masks . nth mask is for index n
    :param indexed_mask:
    :return: n binary masks
    '''
    if n_binaries == None:
        n_binaries = np.max(indexed_mask)
    h,w=binary_masks.shape[0:2]
    concatenated=np.zeros([h,1  ])
    print('conc shape:'+str(concatenated.shape))
    for i in range(n_binaries):
        scaled = binary_masks[:, :, i]*255
        print('scaled shape:'+str(scaled.shape))
        concatenated=np.concatenate((concatenated,scaled),1)
    cv2.imshow('conc',concatenated)
    cv2.waitKey(0)

def generate_images_from_binary_mask(mask,filename,suffix='.png',
                                     max_angle=10,n_angles=2,
                                     max_offset_x=None,n_offsets_x=2,
                                     max_offset_y=None,n_offsets_y=2,
                                     max_scale=1.3,min_scale=0.7,n_scales=2,randomize=True,n_tot=100):
    height=mask.shape[0]
    width=mask.shape[1]
    center = (width/2,height/2)
    eps = 10**-6
    angles = np.arange(-max_angle, max_angle+eps, max_angle*2 / (n_angles-1))
    if max_offset_x == None:
        max_offset_x = int(float(width)/5) #left,right 20% of image
    if max_offset_y == None:
        max_offset_y = int(float(height)/5) #up,down 20% of image
    offsets_x = np.arange(-max_offset_x, max_offset_x+eps, max_offset_x*2/(n_offsets_x-1))
    offsets_y = np.arange(-max_offset_y, max_offset_y+eps, max_offset_y*2/(n_offsets_y-1))
    scales = np.arange(min_scale, max_scale+eps, (max_scale-min_scale)/(n_scales-1))
    print('shape {}\n angles {} \noffsets_x {} \noffsets_y {} \nscales {}'.format(mask.shape,angles,offsets_x,offsets_y,scales))

    mirror_image = cv2.flip(mask,1)
    reflections=[mask,mirror_image]

    n_reflection=-1
    if(0):
        for img_arr in reflections:
            n_reflection=n_reflection+1
            for i in range(n_tot/2):
                angle = np.random.normal(loc=0, scale=max_angle-1.0)
                offset_x = int(np.random.normal(loc=0, scale=max_offset_x))
                offset_y = int(np.random.normal(loc=0, scale=max_offset_x))
                scale = np.random.normal(loc=1.0, scale=max_scale-1.0)
                M = cv2.getRotationMatrix2D(center, angle,scale)
        #                                print('M='+str(M))
                M[0,2]=M[0,2]+offset_x
                M[1,2]=M[1,2]+offset_y
                print('ref {} offx {} offy {} angle {} scale {}'.format(n_reflection,offset_x,offset_y,angle,scale))
        #                        print('M='+str(M))
                xformed_img_arr  = cv2.warpAffine(img_arr,  M, (width,height),borderMode=cv2.BORDER_REPLICATE)
                yield(xformed_img_arr)

    else:
        for img_arr in reflections:
            n_reflection=n_reflection+1
            for offset_x in offsets_x:
                for offset_y in offsets_y:
                    for angle in angles:
                        for scale in scales:
                            theangle = np.random.normal(loc=1.0, scale=max_scale-1.0)

                            M = cv2.getRotationMatrix2D(center, angle,scale)
    #                                print('M='+str(M))
                            M[0,2]=M[0,2]+offset_x
                            M[1,2]=M[1,2]+offset_y
                            print('ref {} offx {} offy {} angle {} scale {}'.format(n_reflection,offset_x,offset_y,angle,scale))
    #                        print('M='+str(M))
                            xformed_img_arr  = cv2.warpAffine(img_arr,  M, (width,height),borderMode=cv2.BORDER_REPLICATE)
                            yield(xformed_img_arr)
    #                                cv2.imwrite(name,xformed_img_arr)
    #                        cv2.imshow('xf',xformed_img_arr)
    #                        cv2.waitKey(0)

def maskname_from_imgname(maskdir,imgname,imgsuffix='.jpg',masksuffix='.png'):
    imgonly=os.path.basename(imgname)
    newname = imgonly.split(imgsuffix)[0]+masksuffix
    newfullpath = os.path.join(maskdir,newname)
#    print('origname {}\nnewname {}\nmaskdir{}'.format(imgname,newfullpath,maskdir))
    return newfullpath


def generate_simultaneous_masks_and_images_dir(imgdir,label_dir,
                                    max_angle=10,n_angles=2,
                                     max_offset_x=10,n_offsets_x=2,
                                     max_offset_y=10,n_offsets_y=2,
                                     max_scale=1.3,min_scale=0.7,n_scales=2):

    imgs =  [os.path.join(image_dir,f) for f in os.listdir(imgdir) if '.jpg' in f]

    for imgname in imgs:
        generate_simultaneous_masks_and_images(imgname,label_dir,
                                        max_angle=max_angle,n_angles=n_angles,
                                         max_offset_x=max_offset_x,n_offsets_x=n_offsets_x,
                                         max_offset_y=max_offset_y,n_offsets_y=n_offsets_y,
                                         max_scale=max_scale,min_scale=min_scale,n_scales=n_scales)

def generate_simultaneous_masks_and_images(imgname,label_dir,
                                    max_angle=10,n_angles=2,
                                     max_offset_x=10,n_offsets_x=2,
                                     max_offset_y=10,n_offsets_y=2,
                                     max_scale=1.3,min_scale=0.7,n_scales=2):

    '''
    :param image_dir:
    :param label_dir:
    :param max_angle:
    :param n_angles:
    :param max_offset_x:
    :param n_offsets_x:
    :param max_offset_y:
    :param n_offsets_y:
    :param max_scale:
    :param min_scale:
    :param n_scales:
    :return:
    '''

    print('imagename:'+imgname)
    img_arr = cv2.imread(imgname)
#    masks = [os.path.join(label_dir,f) for f in os.listdir(label_dir)]
#    maskname = masks[0]
    maskname = maskname_from_imgname(label_dir,imgname)
    print('reading '+maskname)
    mask = cv2.imread(maskname)
    if len(mask.shape)==3:
        print('got 3chan mask')
        mask = mask[:,:,0]
#    mask=mask-1  #fashionista are 1-indexed, rest are not
    binmask = binary_masks_from_indexed_mask(mask, n_binaries=56)

    maskvariations = generate_images_from_binary_mask(binmask,maskname,
                                     max_angle=max_angle,n_angles=n_angles,
                                     max_offset_x=max_offset_x,n_offsets_x=n_offsets_x,
                                     max_offset_y=max_offset_y,n_offsets_y=n_offsets_y,
                                     max_scale=max_scale,min_scale=min_scale,n_scales=n_scales)
    indexed_variations = []

    for variation in maskvariations:
#        show_binary_mask(variation, n_binaries=56)
        indexed = indexed_mask_from_binary_masks(variation)
        indexed_variations.append(indexed)

    origvariations = generate_images_from_binary_mask(img_arr,maskname,
                                     max_angle=max_angle,n_angles=n_angles,
                                     max_offset_x=max_offset_x,n_offsets_x=n_offsets_x,
                                     max_offset_y=max_offset_y,n_offsets_y=n_offsets_y,
                                     max_scale=max_scale,min_scale=min_scale,n_scales=n_scales)

    var_no = 0
    for mask,orig in zip(indexed_variations,origvariations):
#        print('masksize:'+str(mask.shape)+' max:'+str(np.max(mask)))
        newmaskname = maskname.split('.png')[0]+'_var'+str(var_no)+'.png'
#        print('writing new mask to :'+newmaskname)
        cv2.imwrite(newmaskname,mask)
        neworigname = imgname.split('.jpg')[0]+'_var'+str(var_no)+'.jpg'
        print('writing new img to :'+neworigname+', new mask to '+newmaskname)
        cv2.imwrite(neworigname,orig)

            #   show_mask_with_labels(mask_filename,labels,original_image=None,cut_the_crap=False,save_images=False,visual_output=False):

        imutils.show_mask_with_labels(newmaskname,fashionista_categories_augmented_zero_based,visual_output=True,original_image=neworigname)
#        cv2.imshow('orig',orig)
#        cv2.waitKey(0)
        var_no=var_no+1


#==========
def generate_random_pair_mask_and_image_dir(imgdir,label_dir,max_angle=7,max_offset_x=10, max_offset_y=10,
                                     max_scale=1.2,n_tot=3,filter='.jpg',labels=ultimate_21):


    imgfiles = [os.path.join(imgdir,f) for f in os.listdir(imgdir) if filter in f]
  #  imgfiles = imgfiles[0:2]
    print(str(len(imgfiles))+' imagefiles in '+imgdir)
    parallel = True
    if parallel:
        jobs = []
        pool = Pool()
        further_args = [label_dir,{'max_angle':max_angle,'max_offset_x':max_offset_x,'max_offset_y':max_offset_y,'max_scale':max_scale,'n_tot':n_tot,'labels':labels}]

        further_args = [label_dir,{'max_angle':max_angle,'max_offset_x':max_offset_x,'max_offset_y':max_offset_y,'max_scale':max_scale,'n_tot':n_tot,'labels':labels}]
#        x = itertools.izip(imgfiles, itertools.repeat(further_args))
        x = [(img,label_dir) for img in imgfiles]
        print(x)
   #     func_star(x)

        if(1):
            for imgfile in imgfiles:
      #        print(p.map(f, [1, 2, 3]))#

                for i in range(n_tot):
#                     p.map(generate_random_pair_mask_and_image,imgfiles,args=(imgfile,label_dir))
#                     pool.map(generate_random_pair_mask_and_image,*x)
                     pool.map(func_star,x)
                           #,max_angle=max_angle,max_offset_x=max_offset_x,
    #                                                    max_offset_y=max_offset_y,max_scale=max_scale,n_tot=n_tot,labels=labels)
#                    p = multiprocessing.Process(target=generate_random_pair_mask_and_image, args=(imgfile,label_dir,max_angle=max_angle,max_offset_x=max_offset_x,max_offset_y=max_offset_y,max_scale=max_scale,n_tot=n_tot,labels=labels))

               #     p = multiprocessing.Process(target=worker, args=(i,))
                     #jobs.append(pool)
                     #pool.start()

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
#    print('*ab:'+(*a_b))
    return generate_random_pair_mask_and_image(*a_b)

##def main():
#    pool = Pool()
#    a_args = [1,2,3]
#    second_arg = 1
#    pool.map(func_star, itertools.izip(a_args, itertools.repeat(second_arg)))




#    for imgfile in imgfiles:
#        print(p.map(f, [1, 2, 3]))#

#        for i in range(n_tot):
#            generate_random_pair_mask_and_image(imgfile,label_dir,max_angle=max_angle,max_offset_x=max_offset_x,
 #                                               max_offset_y=max_offset_y,max_scale=max_scale,n_tot=n_tot,labels=labels)


def generate_random_pair_mask_and_image(imgname,label_dir,max_angle=7,max_offset_x=20, max_offset_y=20,
                                     max_scale=1.2,labels=ultimate_21):
    '''
    Generate randomly warped img and mask using same params
    :param imgname:
    :param label_dir:
    :param max_angle:
    :param max_offset_x:
    :param max_offset_y:
    :param max_scale:
    :param n_tot:
    :return:
    '''
    global variation_count

    print('imagename:'+imgname)
    img_arr = cv2.imread(imgname)
#    masks = [os.path.join(label_dir,f) for f in os.listdir(label_dir)]
#    maskname = masks[0]
    maskname = maskname_from_imgname(label_dir,imgname)
    print('maskname '+maskname)
    mask = cv2.imread(maskname)
    height=mask.shape[0]
    width=mask.shape[1]
    center = (width/2,height/2)
    if len(mask.shape)==3:
        print('got 3chan mask')
        mask = mask[:,:,0]
#    mask=mask-1  #fashionista are 1-indexed , others not
    binmask = binary_masks_from_indexed_mask(mask, n_binaries=56)
    #randomly flip
    ref = np.random.randint(2)
    reflected=False
    if ref == 1:
        binmask = cv2.flip(binmask,1)
        img_arr = cv2.flip(img_arr,1)
        reflected = True
    angle = np.random.normal(loc=0, scale=max_angle)
    offset_x = int(np.random.normal(loc=0, scale=max_offset_x))
    offset_y = int(np.random.normal(loc=0, scale=max_offset_y))
    scale = np.random.normal(loc=1.0, scale=max_scale-1.0)
    M = cv2.getRotationMatrix2D(center, angle,scale)
#                                print('M='+str(M))
    M[0,2]=M[0,2]+offset_x
    M[1,2]=M[1,2]+offset_y
    print('ref {} offx {} offy {} angle {} scale {} file {}'.format(reflected,offset_x,offset_y,angle,scale,imgname))
#                        print('M='+str(M))
    xformed_mask  = cv2.warpAffine(binmask,  M, (width,height),borderMode=cv2.BORDER_REPLICATE)
    xformed_img_arr  = cv2.warpAffine(img_arr,  M, (width,height),borderMode=cv2.BORDER_REPLICATE)

    indexed_xformed_mask = indexed_mask_from_binary_masks(xformed_mask)
    newmaskname = maskname.split('.png')[0]+'_var'+str(variation_count)+'.png'
    cv2.imwrite(newmaskname,indexed_xformed_mask)
    neworigname = imgname.split('.jpg')[0]+'_var'+str(variation_count)+'.jpg'
    print('writing new img to :'+neworigname+', new mask to '+newmaskname)
    cv2.imwrite(neworigname,xformed_img_arr)
#    imutils.show_mask_with_labels(newmaskname,labels,visual_output=True,original_image=neworigname)
    variation_count = variation_count + 1



global variation_count  #is this possible with a generator?
variation_count = 0

if __name__=="__main__":
    print('running main')
    image  = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test/91692.jpg'
    image_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test'
    label_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_u21'
    label_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels'

    generate_random_pair_mask_and_image_dir(image_dir,label_dir,max_angle=7,max_offset_x=20, max_offset_y=20,
                                     max_scale=1.2,n_tot=100,filter='.jpg',labels=ultimate_21)


#    for i in range(0,10):
#        generate_random_pair_mask_and_image(image,label_dir,max_angle=7,max_offset_x=10, max_offset_y=10,
#                                         max_scale=1.2,n_tot=100)

#    generate_simultaneous_masks_and_images_dir(image_dir,label_dir,
#                            max_angle = 10,n_angles=2,
#                            max_offset_x = 10,n_offsets_x=2,
#                            max_offset_y = 10, n_offsets_y=2,
#                            max_scale=1.2, min_scale=0.8,n_scales=2           )#



#    generate_images_for_directory(image_dir,
#                    max_angle = 10,n_angles=2,
#                    max_offset_x = 10,n_offsets_x=2,
#                    max_offset_y = 10, n_offsets_y=2,
#                    max_scale=1.3, n_scales=2,
#                    noise_level=0.1,noise_type='gauss',n_noises=0,
#                    max_blur=5, n_blurs=0,
#                    do_mirror_lr=True,do_mirror_ud=False,do_bb=False,suffix='.jpg')

#    generate_images_for_directory(label_dir,
#                    max_angle = 10,n_angles=2,
#                    max_offset_x = 10,n_offsets_x=2,
#                    max_offset_y = 10, n_offsets_y=2,
#                    max_scale=1.3, n_scales=2,
 #                   noise_level=0.1,noise_type='gauss',n_noises=0,
  #                  max_blur=5, n_blurs=0,
#                    do_mirror_lr=True,do_mirror_ud=False,do_bb=False,suffix='.png')


#    generate_images(img_filename, max_angle = 3,n_angles=2,
#                    max_offset_x = 50,n_offsets_x=2,
#                   max_offset_y = 50, n_offsets_y=2,
#                   max_scale=1.2, n_scales=2,
#                   noise_level=0.1,noise_type='gauss',n_noises=2,
#                    max_blur=5, n_blurs=2,
#                    do_mirror_lr=True,do_mirror_ud=False,output_dir='snorb')