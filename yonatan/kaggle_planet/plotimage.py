import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def plotImage(outFile):
    '''
    Plot the three color bands for an image

    Automatically detect the number of color bands the 
    image has, and plot the image so that it is easy to 
    visualize all the color bands
    '''

    fileName = "/home/yonatan/Downloads/train-jpg/train_12904.jpg"

    print fileName


    img      = io.imread(fileName)
    numBands = img.shape[-1]

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