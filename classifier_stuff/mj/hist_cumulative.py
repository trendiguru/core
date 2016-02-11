import cv2
import numpy as np
from trendi import Utils

def cumulative_hist(dir, n_bins=[255,255,255],type='lab',mask=None):
    files = Utils.get_files_from_dir_and_subdirs(dir)
    if mask is None or cv2.countNonZero(mask) == 0:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        print "mask shape: " + str(mask.shape)
        print "image shape: " + str(img.shape)
        print str(mask.shape[0] / float(mask.shape[1])) + ',' + str(img.shape[0] / float(img.shape[1]))
        raise ValueError('trouble with mask size, resetting to image size')
    n_pixels = cv2.countNonZero(mask)

    for file in files:
        img = cv2.imread(file)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
        # histograms
        hist_hue = cv2.calcHist([hsv], [0], mask, [bins[0]], [0, 180])
        hist_hue = [item for sublist in hist_hue for item in sublist]  # flatten nested
        hist_hue = np.divide(hist_hue, n_pixels)

        hist_sat = cv2.calcHist([hsv], [1], mask, [bins[1]], [0, 255])
        hist_sat = [item for sublist in hist_sat for item in sublist]
        hist_sat = np.divide(hist_sat, n_pixels)

        hist_int = cv2.calcHist([hsv], [2], mask, [bins[2]], [0, 255])
        hist_int = [item for sublist in hist_int for item in sublist]  # flatten nested list
        hist_int = np.divide(hist_int, n_pixels)
