import cv2
import numpy
import pylab
import numpy
import pickle
import matplotlib.pyplot as plt


img_array = cv2.imread('static.gofugyourself.com_uploads_2014_08_halle-berry-emmys-2014-454167512.png')

s_mat=numpy.float64(numpy.full(( img_array.shape[0],  img_array.shape[1]), 200))
v_mat=s_mat
h_mat= numpy.float64(img_array[:,:,0])
h_mat=h_mat*180./h_mat.max()

hsv_mat=cv2.merge((h_mat,s_mat,v_mat))
hsv_mat=numpy.uint8(hsv_mat)
bgr_image = cv2.cvtColor(hsv_mat, cv2.COLOR_HSV2BGR)

labels=pickle.load( open( "static.gofugyourself.com_uploads_2014_08_halle-berry-emmys-2014-454167512.lbls", "rb" ) )

flat_hue=img_array[:,:,0].ravel()

for i in set(flat_hue):
    pix=numpy.uint8(numpy.asarray([[[i*180/flat_hue.max(),200,200]]]))	
    bgr_color = cv2.cvtColor(pix, cv2.COLOR_HSV2BGR)
    bgr_color=bgr_color/255.
    bgr_color=bgr_color[0][0]
    plt.plot(0,0, color=bgr_color, label=labels.keys()[labels.values().index(i)], linewidth=4) #labels.values()

plt.imshow(bgr_image)
plt.legend(bbox_to_anchor=(1.5, 1), borderaxespad=0.)
plt.show()


