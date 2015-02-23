__author__ = 'jeremy'
import fingerprint_core
import matplotlib.pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilename
import cv2
import numpy as np

def get_file():

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filename)
    img_array=cv2.imread(filename)
 #   cv2.imshow('image',img_array)
 #   cv2.waitKey(100)
    return(img_array)

def fp(img, bounding_box=None):
    if (bounding_box is not None) and (bounding_box != np.array([0, 0, 0, 0])).all():
        img = crop_image_to_bb(img, bounding_box)
    #crop out the outer 1/s of the image for color/texture-based features
    s = 5
    h = img.shape[1]
    w = img.shape[0]
    r = [h / s, w / s, h - 2 * h / s, w - 2 * w / s]

    roi = np.zeros((r[3], r[2], 3), np.uint8)
    # should use imageop.crop here instead, its prob. faster
    for xx in range(r[2]):
        for yy in range(r[3]):
            roi[yy, xx, :] = img[yy + r[1], xx + r[0], :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    #histograms
    bins = 10
    n_pixels = roi.shape[0] * roi.shape[1]

    hist_hue = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_hue = [item for sublist in hist_hue for item in sublist]  #flatten nested
    hist_hue = np.divide(hist_hue, n_pixels)

    hist_sat = cv2.calcHist([hsv], [1], None, [bins], [0, 255])
    hist_sat = [item for sublist in hist_sat for item in sublist]
    hist_sat = np.divide(hist_sat, n_pixels)

    hist_int = cv2.calcHist([hsv], [2], None, [bins], [0, 255])
    hist_int = [item for sublist in hist_int for item in sublist]  #flatten nested list
    hist_int = np.divide(hist_int, n_pixels)

    #Uniformity  t(5)=sum(p.^ 2);
    hue_uniformity = np.dot(hist_hue, hist_hue)
    sat_uniformity = np.dot(hist_sat, hist_sat)
    int_uniformity = np.dot(hist_int, hist_int)

    #Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps = 1e-15
    l_hue = np.log2(hist_hue + eps)
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = np.log2(hist_sat + eps)
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = np.log2(hist_int + eps)
    int_entropy = np.dot(hist_int, l_int)


    fig = plt.figure()
    fig.subplots_adjust(left=0.2, wspace=0.6)

    ax1 = fig.add_subplot(311)
    hist_max=180.0
    bins_start= int(hist_max/float(bins)/2)
    bins_end = int(hist_max-hist_max/float(bins)/2)+1
    print('bins start'+str(bins_start))
    bin_edges = range(0,int(hist_max)+1,int(hist_max/bins))
    print('bin edges'+str(bin_edges))
    bin_centers = range(bins_start,bins_end,int(hist_max/float(bins)))
    print(bin_centers)
    print(len(bin_centers))
    ax1.bar(bin_centers,hist_hue,width=5)
  #  n, bins, patches = plt.hist(hist_hue, num_bins, normed=1, facecolor='green', alpha=0.5)

    ax2 = fig.add_subplot(312)
    ax2.set_title('ylabels aligned')
    ax2.set_ylabel('aligned 1')
  #  ax2.yaxis.set_label_coords(0.1, 0.5)
  #  ax2.set_ylim(0, 2000)
    hist_max=255
    bin_centers = range(hist_max/bins/2,hist_max-hist_max/bins/2,hist_max/bins)
    ax2.bar(bin_centers,hist_sat,width=5)

    ax3 = fig.add_subplot(313)
    hist_max=255
    bin_centers = range(hist_max/bins/2,hist_max-hist_max/bins/2,hist_max/bins)
    ax3.bar(bin_centers,hist_int,width=5)


    fig2 = plt.figure()
#    plt.imshow(img)
#    plt.show()
    cv2.imshow('image',img_array)
    cv2.waitKey(0)


###################################

img_array=get_file()
fp(img_array)

#hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
#histograms
#bins = 25
#n_pixels = h * w
#xs = range(0,180,(180/bins))
#hist_hue = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
#hist_sat = cv2.calcHist([hsv], [1], None, [bins], [0, 255])
#hist_int = cv2.calcHist([hsv], [2], None, [bins], [0, 255])
#Uniformity  t(5)=sum(p.^ 2);
#hue_uniformity = np.dot(hist_hue, hist_hue)
#l_hue = np.log2(hist_hue + eps)
#hue_entropy = np.dot(hist_hue, l_hue)
#l_sat = np.log2(hist_sat + eps)

 #   result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
 #   result_vector = np.concatenate((result_vector, hist_hue, hist_sat), axis=0)