__author__ = 'yonatanlevin'

"""
this function gets three bounding boxes of the same object
which where selected by the SA workers and returns the best
input : BBlist - a list which contains three lists with the BB coordinates
output: BBB - a list with the best coordinates
              if there is no match - None is returned
printouts: the number of matches is printed to log
"""

import math

import cv2
import numpy as np

import background_removal


def selectBest(bblist, imgurl):
    # TODO : read img from url
    largeimg = cv2.imread(imgurl)
    img, ratio = background_removal.standard_resize(largeimg, 400)
    height, width, ch = img.shape
    imgsize = height * width

    # calcuale center of mass per selection
    COM = []
    for i in bblist:
        COM.append((i[0] + i[2] / 2, i[1] + i[3] / 2))

    weights = [0, 0, 0]
    newpoints = []
    total_weight = 0
    simcount = 0

    def minmax(xlist, ylist, wlist, hlist):

        x = int(np.mean(xlist))
        y = int(np.mean(ylist))
        w = int(np.mean(wlist))
        h = int(np.mean(hlist))
        return [x, y, w, h]

    def is_similar(com1, com2, x, y, wid, hig):
        """
        TODO = enter explanation
        :param com1:
        :param com2:
        :param wid:
        :param hig:
        :return:
        """

        oclid_dis_thres = math.sqrt(pow(min(wid), 2) + pow(min(hig), 2))
        oclid_dis_com = math.sqrt(pow(com1[0] - com2[0], 2) + pow(com1[1] - com2[1], 2))
        # overlap
        larger_x_rightcorner = max(x[0], x[1])
        larger_y_rightcorner = max(y[0], y[1])
        smaller_x_leftcorner = min(x[0] + wid[0], x[1] + wid[1])
        smaller_y_leftcorner = min(y[0] + hig[0], y[1] + hig[1])

        overlap = (smaller_x_leftcorner - larger_x_rightcorner) * (smaller_y_leftcorner - larger_y_rightcorner)
        size1 = wid[0] * hig[0]
        size2 = wid[1] * hig[1]

        if oclid_dis_com < 0.5 * oclid_dis_thres:
            # TODO : decide the dist percent
            # check overlap
            weight = 2.0 * overlap / (size1 + size2)
            return weight

        return 0

    for f in range(0, 3, 1):
        if f == 0:
            num1 = 0
            num2 = 1
        elif f == 1:
            num1 = 0
            num2 = 2
        else:
            num1 = 1
            num2 = 2

        x = [bblist[num1][0], bblist[num2][0]]
        y = [bblist[num1][1], bblist[num2][1]]
        wid = [bblist[num1][2], bblist[num2][2]]
        hig = [bblist[num1][3], bblist[num2][3]]

        # size threshold check - selection bigger than 1% of original img size
        if min(wid[0] * hig[0], wid[1] * hig[1]) > 0.01 * imgsize:
            weights[f] = is_similar(COM[num1], COM[num2], x, y, wid, hig)
            if weights[f] > 0.2:
                simcount += 1
                total_weight += weights[f]
            else:
                weights[f] = 0

        newpoints.append(minmax(x, y, wid, hig))

    if simcount == 0:
        print 'WARNING : three different entries'
        return None

    if simcount == 1:
        print 'one similar'

    if simcount == 2:
        print 'two similar'

    if simcount == 3:
        print 'three similar'

    newx = int(math.floor(weights[0] / total_weight * newpoints[0][0] +
                          weights[1] / total_weight * newpoints[1][0] +
                          weights[2] / total_weight * newpoints[2][0]))
    newy = int(math.floor(weights[0] / total_weight * newpoints[0][1] +
                          weights[1] / total_weight * newpoints[1][1] +
                          weights[2] / total_weight * newpoints[2][1]))
    neww = int(math.floor(weights[0] / total_weight * newpoints[0][2] +
                          weights[1] / total_weight * newpoints[1][2] +
                          weights[2] / total_weight * newpoints[2][2]))
    newh = int(math.floor(weights[0] / total_weight * newpoints[0][3] +
                          weights[1] / total_weight * newpoints[1][3] +
                          weights[2] / total_weight * newpoints[2][3]))

    return [newx, newy, neww, newh]
