__author__ = 'yuli'

import Utils
import background_removal
import kassper
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_fp(image_url, bb=None):
    image = Utils.get_cv2_img_array(image_url)
    small_image, resize_ratio = background_removal.standard_resize(image, 400)
    mask = get_mask(small_image )
    print mask.shape
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    fp_vector = yuli_fp(small_image, mask, whaterver=63)
    return fp_vector

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

def yuli_fp(small_image, mask, whaterver=45):
	# Write awesome function here
    from itertools import product, chain
    import MR8filters

    sigmas = [1, 2, 4]
    n_sigmas = len(sigmas)
    n_orientations = 6

    edge, bar, rot = MR8filters.makeRFSfilters(sigmas=sigmas,
            n_orientations=n_orientations)

    n = n_sigmas * n_orientations

    #multiply small_img with mask
    img = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    mask = mask
    img = np.multiply(mask, img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    img = img.astype(np.float)
    #apply filters
    filterbank = chain(edge, bar, rot)
    n_filters = len(edge) + len(bar) + len(rot)
    response = MR8filters.apply_filterbank(img, filterbank)
    #fp_vector = MRFfilters(small_image, mask)

    # plot responses

    # plot responses
    fig2, ax2 = plt.subplots(3, 3)
    for axes, res in zip(ax2.ravel(), response):
        axes.imshow(res, cmap=plt.cm.gray)
        axes.set_xticks(())
        axes.set_yticks(())
    ax2[-1, -1].set_visible(False)
    plt.show()

    return response #return fp_vector


if __name__ == "__main__":
    run_fp('10796178-sexy-short-gown.jpg')

