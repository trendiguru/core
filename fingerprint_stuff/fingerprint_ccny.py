from __future__ import division #make division floating point not integer
# __author__ = 'liorsabag'
import matplotlib as plt
import numpy as np
import cv2
import string
import logging
import time
from pylab import *



def crop_image_to_bb(img, bb_coordinates_string_or_array):
    if isinstance(bb_coordinates_string_or_array, basestring):
        bb_array = [int(bb) for bb in string.split(bb_coordinates_string_or_array)]
    else:
        bb_array = bb_coordinates_string_or_array

    x = bb_array[0]
    y = bb_array[1]
    w = bb_array[2]
    h = bb_array[3]
    hh, ww, d = img.shape
    if (x + w <= ww) and (y + h <= hh):
        #rectok=True
        r = [x, y, w, h]
        #allRects.append(r)
        mask = np.zeros(img.shape[:2], np.uint8)
        cropped_img = np.zeros((r[3], r[2], 3), np.uint8)
        mask[r[0]:r[2], r[1]:r[3]] = 255

        for xx in range(r[2]):
            for yy in range(r[3]):
                cropped_img[yy, xx, :] = img[yy + r[1], xx + r[0], :]

    else:
        cropped_img = img
        logging.warning('Could not crop. Bad bounding box: imsize:' + str(ww) + ',' + str(hh) +
                        ' vs.:' + str(x + w) + ',' + str(y + h))

    return cropped_img

from Tkinter import Tk
from tkFileDialog import askopenfilename

def get_file():

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filename)
    img_array=cv2.imread(filename)
 #   cv2.imshow('image',img_array)
 #   cv2.waitKey(100)
    return(img_array)



def find_color_percentages(img_array):
    """
    find colors using hue quantized into a few sectors, plus white , black, and grey

    """


    white_saturation_max = 36  # maximum S value for a white pixel14 from the gimp , 14*100/255
    white_value_min = 214 #  minimum V value for a white pixel this is 84 * 100/255
    black_value_max = 23 # maximum V value for a black pixel, 15*100/255
    n_colors=10
    color_limits=range(0,180+int(180/n_colors),int(180/n_colors))
    #print(color_limits)

    white_count=0
    black_count=0
    grey_count=0

    mhue=0
    mval=0
    msat=0

    t0=time.time()
    h_arr=hsv[:,:,0]
    s_arr=hsv[:,:,1]
    v_arr=hsv[:,:,2]
 #ways to count array elements fitting thresholds
 #    np.sum(myarray >= thresh)
#np.size(np.where(np.reshape(myarray,-1) >= thresh))
#fast array way to do same calculation

 #slow pixel by pixel way

#    t0=time.time()
#    for x in range(r[2]):
#        for y in range(r[3]):
#            mhue=hsv[x,y, 0]   #
#            msat=hsv[x,y, 1]   #
#            mval=hsv[x,y, 2]   #the hsv values of current pixel
#     #       print('x,y:'+str(x)+','+str(y)+' hue:'+str(mhue)+' val:'+str(mval)+' sat:'+str(msat))
#            if mval<black_value_max:
#                black_count+=1
#            elif msat<white_saturation_max:
 #               if mval>white_value_min:
 #                  white_count+=1
 #              else:
 #                   grey_count+=1
 #  t1=time.time()
 #  print('whitecount:'+str(white_count)+ 'greycount:'+str(grey_count)+' blackcount:'+str(black_count)+' dt:'+str(t1-t0)+' area'+str(area))#

    mask[0] = image[0]==0
    mask[1] = image[1]==0
    mask[2] = image[2]==0

    masksofi = mask[0] * mask[1] * mask[2]

    black_count=np.sum(v_arr<black_value_max)
    black_percentage=black_count/area


    white_mask=(s_arr<white_saturation_max) *(v_arr>white_value_min)




    white_count=np.sum(white_mask)
    white_percentage=white_count/area
    grey_count=np.sum((s_arr<white_saturation_max) *( v_arr<=white_value_min) *( v_arr>=black_value_max))
    grey_percentage=grey_count/area
    inv=np.invert(white_mask)
    color_mask=(np.invert(white_mask))*(v_arr>=black_value_max)
    colors_count=np.sum(color_mask)
    print("tot color count:"+str(tot_colors))
    color_counts=[]
    for i in range(0,n_colors):
        color_percentages.append(np.sum(  color_mask*(h_arr<color_limits[i+1])*(h_arr>=color_limits[i])))
        if DEBUG:
            print('color '+str(i)+' count ='+str(color_percentages[i]))
            print('color percentages:'+str(color_percentages))
        color_percentages[i]=color_percentages[i]/area  #turn pixel count into percentage
    all_colors=np.zeros(3)
    all_colors[0]=white_percentage
    all_colors[1]=black_percentage
    all_colors[2]=grey_percentage
    all_colors=np.append(all_colors,color_counts)

    #   all_colors=np.concatenate(all_colors,color_counts)
    if DEBUG:
        print('white black grey colors:'+str(all_colors))   #order is : white, black, grey, color_count[0]...color_count[n_colors]
        print('sum:'+str(np.sum(all_colors)))
 #   all_colors=color_counts
 #   np.append(all_colors,white_count)
 #   np.append(all_colors,black_count)
 #   all_colors.append(grey_count)

    #dominant_color_indices, dominant_colors = zip(*sorted(enumerate(all_colors), key=itemgetter(1), reverse=True))
    #above is for array, now working with numpy aray

# the order of dominant colors is what ccny guys used, if we just have vector in order of color i think its just as good
#so for now the following 3 lines are not used
    dominant_color_indices=np.argsort(all_colors, axis=-1, kind='quicksort', order=None)
    dominant_color_indices = dominant_color_indices[::-1]
    dominant_color_percentages=np.sort(all_colors, axis=-1, kind='quicksort', order=None)
    dominant_color_percentages = dominant_color_percentages[::-1]

    if DEBUG:
        print('color percentages:'+str(dominant_color_percentages)+' indices:'+str(dominant_color_indices))
    t1=time.time()
    return(all_colors)

def fp_ccny(img, bounding_box=None):
    """

    :param img:
    :param bounding_box:
    :return:
    takes image, box
    returns numpy array - the fingerprint of the image, as done in ccny paper

        """

    global DEBUG
    DEBUG=True
    show_visual_output=True
    if (bounding_box is not None) and (bounding_box != np.array([0, 0, 0, 0])).all():
        img = crop_image_to_bb(img, bounding_box)
    #crop out the outer 1/s of the image for color/texture-based features
    s = 5
    h, w, depth = img.shape
    r = [int(h /s), int(w /s), h - 2 * int(h /s), w - 2 * int(w /s)]
    area=(h - 2 * int(h / s))*( w - 2 * int(w /s))  #this is the area after cropping off s from each side
    if DEBUG:
            print('image  size:'+str(w)+'x'+str(h)+'x'+str(depth))
            print('croppedsize:'+str(r(2)-r(0))+'x'+str(r(3)-r(1))+'x'+str(depth)+' area:'+str(area))

    roi = np.zeros((r[2], r[3], 3), np.uint8)
    for xx in range(r[2]):
        for yy in range(r[3]):
            roi[xx, yy, :] = img[ xx + r[0],yy + r[1], :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    if show_visual_output is True:
        cv2.imshow('image',img)
        cv2.imshow('roi',roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    color_percentages = find_color_percentages(hsv)
#    print('whitecount:'+str(white_count)+ ' greycount:'+str(grey_count)+' blackcount:'+str(black_count)+' dt:'+str(t1-t0)+' area:'+str(area) %white_count)
    #OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    #histograms
    bins = 25
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

    n_color_elements=3
    result_vector=np.zeros(shape=(0))

    for i in range(0,n_color_elements):
 #       print(i)
        result_vector=np.append(result_vector,float(dominant_color_indices[i]))
    for i in range(0,n_color_elements):
        result_vector=np.append(result_vector,dominant_color_percentages[i])
    #more_elements=[hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    np.append(result_vector,hue_uniformity)
    np.append(result_vector,sat_uniformity)
    np.append(result_vector,int_uniformity)
    np.append(result_vector,hue_entropy)
    np.append(result_vector,sat_entropy)
    np.append(result_vector,int_entropy)
                    #    result_vector=result_vector+more_elements
    #result_vector = result_vector+np.concatenate((result_vector, hist_hue, hist_sat), axis=0)

   # result_vector=np.concatenate(result_vector,hist_hue)
   #readd
    print(result_vector)

    return result_vector


def nothing(x):
    pass

def make_sliders(options):
# Create a black image, a window

    cv2.namedWindow('image')
   # print(options)
    # create trackbars for color change
    for item in options:
        print(item)
        for key in item:
            if item[key] in ["Boolean","boolean","Bool","bool"]:
                range_min=0
                range_max=1
                switch = '0 : OFF \n1 : ON'
                cv2.createTrackbar(switch, 'image',0,1,nothing)
            else:
                range_min=item[key][0]
                range_max=item[key][1]
                cv2.createTrackbar(key,'image',range_min,range_max,nothing)
#                cv2.createTrackbar("H-Min", "Processed", processor.tmin1, 255, processor.min1)
            print('varname:'+key+' min:'+str(range_min)+' max:'+str(range_max))

    # create switch for ON/OFF functionality
#    switch = '0 : OFF \n1 : ON'
#    cv2.createTrackbar(switch, 'image',0,1,nothing)



def get_params(options):
    vals=[]
   # print(options)
    for item in options:
        print(item)
        for key in item:

            if item[key] in ["Boolean","boolean","Bool","bool"]:
                switch = '0 : OFF \n1 : ON'
                value = cv2.getTrackbarPos(switch,'image')
                print('varname:'+key+' val:'+str(value))

            else:
                value = cv2.getTrackbarPos(key,'image')
    #        print('varname:'+key+' val:'+str(value))
            vals.append(value)
    return(vals)


def fp_old(img, bounding_box=None):
    if (bounding_box is not None) and (bounding_box != np.array([0, 0, 0, 0])).all():
        img = crop_image_to_bb(img, bounding_box)
    #crop out the outer 1/s of the image for color/texture-based features
    s = 5
    h = img.shape[1]
    w = img.shape[0]
    r = [h / s, w / s, h - 2 * h / s, w - 2 * w / s]

    roi = np.zeros((r[3], r[2], 3), np.uint8)
    for xx in range(r[2]):
        for yy in range(r[3]):
            roi[yy, xx, :] = img[yy + r[1], xx + r[0], :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    #histograms
    bins = 25
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
    #watch out! this eps choice affects the results!!!
    eps = 1e-15
    l_hue = np.log2(hist_hue + eps)
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = np.log2(hist_sat + eps)
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = np.log2(hist_int + eps)
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    result_vector = np.concatenate((result_vector, hist_hue, hist_sat), axis=0)

    return result_vector

def unwrinkle(img_array,params):
    """
    :param img_array:
    :return unwrinkled image:
    """
    print('unwrinkling...')

    #greyscale
    #gaussian smooth
    #canny edge
    #morphology (dilate i guess) to remove small ad edges
    #histogram equalization
    #select circle centered on image
    #radon xform
    #rotate to 0
    #haar wavelet
    #co-occurence matrix for wavelet subimages
    #six statistical features


#    h, w = img_array.shape[:2]
#maybe deal with grayscale input images.......tomorrow
#defaults
#    blur_kernelsize=1
#    blur_sigma=9
#    edge_minval=30
#    edge_maxval=90
#    edge_aperture_size=3
#    use_accurate_gradient=1
#    erode_size=1

  #  print locals().keys()
    blur_kernelsize=params[0]
    blur_sigma=params[1]
    edge_minval=params[2]
    edge_maxval=params[3]
    edge_aperture_size=params[4]
    use_accurate_gradient=params[5]
    erode_size=params[6]

#convert to gray
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

#blur (guassian) kernelsize has to be odd
    if blur_kernelsize%2 != 1 :
        blur_kernelsize=blur_kernelsize+1
    blurred=cv2.GaussianBlur(gray,(blur_kernelsize,blur_kernelsize),blur_sigma)

#find edges - canny edge detector
#    edges = cv2.Canny(blurred,minVal=edge_minval,maxVal=edge_maxval,aperture_size=edge_aperture_size,L2gradient=use_accurate_gradient)
    #edge_map=
    if edge_aperture_size % 2 != 1:
        edge_aperture_size=edge_aperture_size+1
    if edge_aperture_size <3:
        edge_aperture_size = 3    #edge_map=
    if edge_aperture_size >7:
        edge_aperture_size = 7

    if edge_minval is 0:
        edge_minval = 1
    if edge_maxval is 0:
        edge_maxval = 1
  #  edge_minval=50
 #   edge_maxval=100
    if use_accurate_gradient:
        edges = cv2.Canny(image=blurred,threshold1=edge_minval,threshold2=edge_maxval,apertureSize=edge_aperture_size,L2gradient=True)
    else:
  #$      edges = cv2.Canny(image=blurred,threshold1=edge_minval,threshold2=edge_maxval,apertureSize=edge_aperture_size,L2gradient=False)size
        print('eap:'+str(edge_aperture_size))
        edges = cv2.Canny(image=blurred,threshold1=edge_minval,threshold2=edge_maxval,apertureSize=edge_aperture_size,L2gradient=False)
#erod
    if erode_size<1:
        erode_size=1
#    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(erode_size,erode_size))
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erode_size,erode_size))
    eroded = cv2.erode(edges,element)

    return(eroded)

import os
def my_hough_lines(img_array):
    gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize=3)

    print('ok0')
    rho=2
    lines = cv2.HoughLines(edges,rho,np.pi/180,200)

    print('ok1')
    lines = cv2.HoughLinesWithAccumulator(edges,rho,np.pi/180,200)
    print('ok2')
    lines, arr = cv2.HoughLinesWithAccumulator(edges,rho,np.pi/180,200)

    print('shape:'+str(arr.shape)+' size:'+str(arr.size)+' dtype:'+str(arr.dtype))
    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

        cv2.line(img_array,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow('lines',img_array)
        k = cv2.waitKey(0) & 0xFF

    print('still ok1')

    if arr is not None:
        minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(arr)
        print('min:'+str(minVal)+' max:'+str(maxVal)+' minloc:'+str(minLoc)+'maxLoc:'+str(maxLoc))
        res = arr*255.0/maxVal
        res2 = np.uint8(res)
        arr_equalized=cv2.equalizeHist(res2)
        cv2.imshow('arr',arr_equalized)
        k = cv2.waitKey(0) & 0xFF


def do_hardcoded():
    blur_kernelsize=1
    blur_sigma=9
    edge_minval=30
    edge_maxval=90
    edge_aperture_size=3
    use_accurate_gradient=1
    erode_size=1

    import os
    path="testimages"
        #os.path.join("/home","mypath","to_search")
    for r,d,f in os.walk(path):
        for files in f:
           if files[-3:].lower()=='jpg' or files[-4:].lower() =="jpeg":
                filename=os.path.join(r,files)
                print "found: ",filename
                img_array=cv2.imread(filename)
                my_hough_lines(img_array)
                params=(1,9,30,90,3,1,1)
                print(params)
                unwrinkled=unwrinkle(img_array,params)
#        cv2.namedWindow('image')
#        cv2.namedWindow('orig')
                cv2.imshow('unwrinkled',unwrinkled)
                cv2.imshow('original',img_array)
                k = cv2.waitKey(1000) & 0xFF
                #cv2.destroyAllWindows()


############### ########333
#test the various functions
#################

do_hardcoded()

img_array=get_file()
#print(str(img_array[10,1:]))
#plt.figure()
options=[{"blur_kernelsize":(0,100)},{"blur_sigma":(0,100)},{"edge_minval":(0,300)}, \
         {"edge_maxval":(0,300)},{"edge_aperture_size":(3,7)},{"use_accurate_gradient":"Boolean"},\
         {"erode_size":(1,100)}]
#cv2.namedWindow('image')
# Show an image in the window
#cv2.imshow('image',img_array)
make_sliders(options)
params=get_params(options)



#plt.figure('figure1')
#imgplot = imshow(img_array)
#imgplot.set_cmap('spectral')

#plt.imshow(img_array)
#plt.show()
#unwrinkled=unwrinkle(img_array,blur_kernelsize=(3,3),blur_sigma=4,edge_minval=50,edge_maxval=200,edge_aperture_size=3,use_accurate_gradient=True)
#def unwrinkle(img_array,blur_kernelsize=(5,5),blur_sigma=5,edge_minval=100,edge_maxval=200,edge_aperture_size=3,use_accurate_gradient=True):

#    Python: cv2.setTrackbarPos(trackbarname, winname, pos)-> None

while(1):

    params=get_params(options)
#    blur_kernelsize=params["blur_kernelsize"]
#    blur_sigma=params["blur_sigma"]
#    edge_minval=params["edge_minval"]
#    edge_maxval=params["edge_maxval"]
#    edge_aperture_size=params["edge_aperture_size"]
    print('params:'+str(params))
    unwrinkled=unwrinkle(img_array,params)
    cv2.imshow('image',unwrinkled)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
    cv2.imshow('original',img_array)



'''
we first detect colors of white, black, and gray based on saturation S
and luminance I. If the luminance I of a pixel is large enough, and saturation S is less than a special
threshold, then we define the color of the pixel as white. Similarly, the color of a pixel black, can
be determined if the luminance I of a pixel is less enough and saturation S is also satisfied with the
condition. Under the rest values of the luminance I, pixel of color gray could be found in a defined
small S radius range. For other colors (e.g. red, orange, yellow, green, cyan, blue, purple, and pink),
hue information is employed
'''