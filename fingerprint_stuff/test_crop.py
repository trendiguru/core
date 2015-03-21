__author__ = 'jeremy'
import cv2
import fingerprint_core
import background_removal
import Utils

im=Utils.get_cv2_img_array('http://www.scot-image.co.uk/stock/gallery/North%20Cyprus/various/Northern-Cyprus-goats-Z0219015.jpg', download = True, try_url_locally=True,download_directory='images')
cv2.imshow('orig',im)
cropped = fingerprint_core.crop_image_to_bb(im,[10,150,100,200])
cv2.imshow('cropped',cropped)
cv2.waitKey(0)