__author__ = 'yonatanlevin'

"""
this function gets three bounding boxes of the same object
which where selected by the click workers and returns the best
input : BBlist - a list which contains three lists with the BB coordinates
output: BBB - a list with the best coordinates
              if there is no match - None is returned
printouts: the number of matches is printed to log
"""

import math

import numpy as np

import utils_tg
import background_removal


def selectBest(bblist, imgurl):
    large_img = utils_tg.get_cv2_img_array(imgurl)
    img, ratio = background_removal.standard_resize(large_img, 400)
    height, width = img.shape[0:2]
    img_size = height * width

    # calcuale center of mass per selection
    COM = []
    for i in bblist:
        COM.append((i[0] + i[2] / 2, i[1] + i[3] / 2))

    weights = [0, 0, 0]
    new_points = []
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

    couples = [(0, 1), (0, 2), (1, 2)]
    for idx, pair in enumerate(couples):
        x = [bblist[pair[0]][0], bblist[pair[1]][0]]
        y = [bblist[pair[0]][1], bblist[pair[1]][1]]
        wid = [bblist[pair[0]][2], bblist[pair[1]][2]]
        hig = [bblist[pair[0]][3], bblist[pair[1]][3]]

        # size threshold check - selection bigger than 1% of original img size
        if min(wid[0] * hig[0], wid[1] * hig[1]) > 0.01 * img_size:
            weights[idx] = is_similar(COM[pair[0]], COM[pair[1]], x, y, wid, hig)
            if weights[idx] > 0.2:
                simcount += 1
                total_weight += weights[idx]
            else:
                weights[idx] = 0

        new_points.append(minmax(x, y, wid, hig))

    if simcount == 0:
        print 'WARNING : three different entries'
        return None

    if simcount == 1:
        print 'one similar'

    if simcount == 2:
        print 'two similar'

    if simcount == 3:
        print 'three similar'

    para = []
    for p in range(4):
        para.append(int(math.floor(weights[0] / total_weight * new_points[0][p] +
                                   weights[1] / total_weight * new_points[1][p] +
                                   weights[2] / total_weight * new_points[2][p])))

    # para = [x,y,w,h]
    return para
