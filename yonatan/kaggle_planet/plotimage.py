import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
from PIL import Image


def plotImage(outFile):
    '''
    Plot the three color bands for an image

    Automatically detect the number of color bands the 
    image has, and plot the image so that it is easy to 
    visualize all the color bands
    '''

    fileName = "/home/yonatan/Downloads/train-jpg/train_12904.jpg"

    print fileName

    full_image = cv2.imread(fileName)
    # blue, green, red = cv2.split(full_image)
    #
    # print "blue channel: {}".format(blue)
    # print "green channel: {}".format(green)
    # print "red channel: {}".format(red)
    # # print "nir channel: {}".format(nir)

    # nir = full_image[:, :, 2, 0]
    # print "nir channel: {}".format(nir)
    print "full_image: {}".format(full_image)


    img      = io.imread(fileName)
    numBands = img.shape[-1]

    print "img: {}".format(img)

    im = Image.open(fileName)
    im = im.convert("L")
    im = np.asarray(im)

    print "im: {}".format(im)

    # im.show()
    # im.tostring()





    print "numBands: {0}".format(numBands)

    plt.figure(figsize=(5*(numBands+1), 5)) # One for the original image
    f = 1.0/(numBands+1)
    for b in range(numBands):
        plt.axes( [b * f, 0, f, 1] )
        plt.imshow(img[ :, :, b], cmap=plt.cm.viridis)
        plt.xticks([]); plt.yticks([])

    if numBands == 4:
        img1 = img[ :, :, :-1].copy()
        img1 = img1*255.0/img1.max()
        img2 = np.zeros(np.shape(img1))
        img2[:,:,0] = img1[:,:,2]
        img2[:,:,1] = img1[:,:,1]
        img2[:,:,2] = img1[:,:,0]
        img2 = img2.astype(np.uint8)
    else:
        img2 = img

    plt.axes( [(b+1) * f, 0, f, 1] )
    plt.imshow(img2)
    plt.xticks([]); plt.yticks([])

    plt.savefig(outFile)
    plt.close('all')

    return img