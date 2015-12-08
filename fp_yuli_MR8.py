__author__ = 'yuli'

import pickle

import os

import cv2

import numpy as np

import Utils
import background_removal
import kassper


def is_relevant_head(first_head):

    if first_head[2] < 32 or first_head[3] < 32:
        return False
    return True

def resize_by_head(small_image, first_head):
    #resize to min head dimention =32
    d=32
    original_w = small_image.shape[1]
    original_h = small_image.shape[0]

    if first_head[2] < first_head[3]:
        first_head_min = first_head[2]
        resize_ratio = float(d)/first_head_min
        new_w = original_w*resize_ratio
        new_h = original_h*resize_ratio
    else:
        first_head_min = first_head[3]
        resize_ratio = float(d)/first_head_min
        new_w = original_w*resize_ratio
        new_h = original_h*resize_ratio
    resized_image = cv2.resize(small_image, (int(new_w), int(new_h)))
    return resized_image


# Returned mask is resized to max_side_lenth 400
def get_mask(small_image, bb=None):
    if bb is not None:
        bb = [int(b) for b in (np.array(bb) / resize_ratio)]  # shrink bb in the same ratio
    fg_mask = background_removal.get_fg_mask(small_image, bb)                     # returns the grab-cut mask (if bb => PFG-PBG gc, if !bb => face gc)
    gc_image = background_removal.get_masked_image(small_image, fg_mask)
    without_skin = kassper.skin_removal(gc_image, small_image)
    crawl_mask = kassper.clutter_removal(without_skin, 400)
    without_clutter = background_removal.get_masked_image(without_skin, crawl_mask)
    fp_mask = kassper.get_mask(without_clutter)
    return fp_mask

def trim_mask(small_image, mask):
    # returns trimmed mask in grey scale

    img = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    #img = np.multiply(mask, img), min dim
    ind = np.nonzero(mask)
    min_y=np.min(ind[0]);    max_y=np.max(ind[0]) #; ry = np.abs(max_y-min_y)
    min_x=np.min(ind[1] );    max_x=np.max(ind[1] ) #; rx = np.abs(max_x-min_x)
    #min_dim = np.min(np.asarray([rx, ry]))
    #facx = rx/10 ; facy =ry/10 ; centx = min_x+rx/2 ; centy = min_y+ry/2

    trimmed_mask = img[min_y:max_y, min_x:max_x]
    cv2.imwrite('mr8_trimmed.jpg', trimmed_mask)
    
    #print trimmed_mask.shape
    #img = cv2.rectangle(img ,(min_x+facx, min_y+facy),(max_x-facx, max_y-facy),(255, 0, 0))
    #img = cv2.rectangle(img ,(centx-facx, centy-facy),(centx+facx, centy+facy),(255, 0, 0))
    return trimmed_mask


def yuli_fp(trimmed_mask, feat_size):
    from itertools import chain
    import MR8filters

    #sample_texture (trimmed_mask , feature_size= min_dim/10 , n= 1 ):

    centy = trimmed_mask.shape[0]/2 ; centx = trimmed_mask.shape[1]/2
    #img = cv2.rectangle(trimmed_mask ,(centx-feature_size, centy-feature_size),(centx+feature_size, centy+feature_size),(255, 0, 0))
    feature = trimmed_mask[centy - feat_size / 2:centy + feat_size / 2, centx - feat_size / 2:centx + feat_size / 2]
    print "sample shape:", feature.shape
    cv2.imwrite('mr8_samp.jpg',feature)

    d = 40
    if feat_size < d:
	print "samp_size too small !!!"
        return
    feature = cv2.resize(feature, (d, d))
    #returns n=3 texture features

    # Make MR8 filters
    edge, bar, rot = MR8filters.makeRFSfilters()
    feature = feature.astype(np.float)
    # Normalize feature
    feature = (feature - np.mean(feature))/np.std(feature)

    # np.set_printoptions(precision=2)
    # for i in range(len(feature)):
    #     print feature[i]
    # print np.mean(feature)

    # apply filters
    filterbank = chain(edge, bar, rot)
    n_filters = len(edge) + len(bar) + len(rot)
    response = MR8filters.apply_filterbank(feature, filterbank)



#    # plot filters
#    n_sigmas = 3
#    n_orientations = 6
#    # 2 is for bar / edge, + 1 for rot
#    fig, ax = plt.subplots(n_sigmas * 2 + 1, n_orientations)
#    for k, filters in enumerate([bar, edge]):
#        for i, j in product(xrange(n_sigmas), xrange(n_orientations)):
#            row = i + k * n_sigmas
#            ax[row, j].imshow(filters[i, j, :, :], cmap=plt.cm.gray)
#            ax[row, j].set_xticks(())
#            ax[row, j].set_yticks(())
#    ax[-1, 0].imshow(rot[0, 0], cmap=plt.cm.gray)
#    ax[-1, 0].set_xticks(())
#    ax[-1, 0].set_yticks(())
#    ax[-1, 1].imshow(rot[1, 0], cmap=plt.cm.gray)
#    ax[-1, 1].set_xticks(())
#    ax[-1, 1].set_yticks(())
#    for i in xrange(2, n_orientations):
#        ax[-1, i].set_visible(False)


   # plot responses
#   fig2, ax2 = plt.subplots(3, 3)
#    for axes, res in zip(ax2.ravel(), response):
#        axes.imshow(res, cmap=plt.cm.gray)
#        axes.set_xticks(())
#        axes.set_yticks(())
#    plt.subplot(3, 3, 9)
#    #plt.imshow(feature, cmap=plt.cm.gray)
#    #plt.savefig(path+'MR8'+name)
#    #plt.format_coord = format_coord
#    ax2[-1, -1].set_visible(False)
#    #plt.show()

    return response

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def mean_std_pooling(response, s):
    # response dimentions should be mod(n=8)
    result = []
    bl_shaped = blockshaped(response, s, s)
    print bl_shaped.shape

    for block in bl_shaped:
        result.append(np.mean(block))
        if np.isnan(np.mean(block)) == True:
            print 'mean isNaN'

        result.append(np.std(block))
        if np.isnan(np.std(block)) == True:
            print 'std isNaN'
    return np.asarray(result)


if __name__ == "__main__":


    paths = [["/home/omer/tmp_folder/","/home/omer/"]]


    path_in = paths[0][0]
    path_out = paths[0][1]

    listing = os.listdir(path_in)

    responses = []

    for filename in listing:
       # im = Image.open(path1 + file)
        print filename
        image = Utils.get_cv2_img_array(path_in+filename)
        #small_image, resize_ratio = background_removal.standard_resize(image, 400)

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print gray_img.shape
        
        samp_size = 40
        response = yuli_fp(gray_img, filename, path_out , samp_size/2 )
       # print len(response)

        ms_response = []
        for idx, val in enumerate(response):
            ms_response.append(mean_std_pooling(val, 5))

        print (ms_response)
        print "shape: " , ms_response[0].shape
	
        responses.append(ms_response)

    with open(path_out+'Results_MR8.pickle', 'w') as f:
        pickle.dump(responses, f)











