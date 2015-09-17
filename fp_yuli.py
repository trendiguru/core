import Utils
import background_removal
import kassper
import cv2

def run_fp(image_url, bb=None):
	image = Utils.get_cv2_img_array(image_url)
	small_image, resize_ratio = background_removal.standard_resize(image, 400)
	mask = get_mask(small_image, bb)
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
	pass

