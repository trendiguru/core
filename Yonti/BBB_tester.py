__author__ = 'yonatanlevin'

"""
this is a tester for the BBB
inner func:
generate_BB =  func for selecting BB on a given img
displayAll = presents entries & result on the same image
"""

import cv2

import background_removal
import BBB

# choosing a test image : manually or direct path
# testImage = background_removal.get_image()
imgpath = '/Users/yonatanlevin/Downloads/test1.jpg'
testImage = cv2.imread(imgpath)
Img, ratio = background_removal.standard_resize(testImage, 400)

clone = Img.copy()
tmpList = []
selectflag = False
selectlist = []


def displayAll(bestBB, listOf3):
    for l in listOf3:
        tmp = [(l[0], l[1]), (l[0] + l[2], l[1] + l[3])]
        cv2.rectangle(Img, tmp[0], tmp[1], (255, 0, 0), 1)

    if bestBB != [None]:
        tmpBB = [(bestBB[0][0], bestBB[0][1]), (bestBB[0][0] + bestBB[0][2], bestBB[0][1] + bestBB[0][3])]
        cv2.rectangle(Img, tmpBB[0], tmpBB[1], (0, 0, 255), 2)

    cv2.imshow('RESULTS', Img)
    cv2.waitKey(0)


def generate_BB(Img):
    imgTitle = 'Select the Dress'
    global selectlist, selectflag

    def selectBB(event, x, y, flags, param):

        global selectlist, selectflag

        if event == cv2.EVENT_LBUTTONDOWN:
            selectlist = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            selectlist.append((x, y))
            cv2.rectangle(Img, selectlist[0], selectlist[1], (0, 255, 0), 2)
            selectflag = True

    cv2.namedWindow(imgTitle)
    cv2.setMouseCallback(imgTitle, selectBB)

    while True:
        cv2.imshow(imgTitle, Img)
        tmpList = selectlist
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if selectflag:
            print tmpList
            if tmpList[0][0] != tmpList[1][0] and tmpList[0][1] != tmpList[1][1]:
                selectflag = False
                break
            else:
                print 'bad selection'
                selectlist = []
                selectflag = False

    cv2.destroyAllWindows()

    return tmpList


listOf3couples = []
listOf3 = []

for i in range(0, 3, 1):
    listOf3couples.append(generate_BB(Img))

    # set the point from left to right
    if listOf3couples[i][0][0] > listOf3couples[i][1][0]:
        x = listOf3couples[i][0][0]
        y = listOf3couples[i][0][1]
        x1 = listOf3couples[i][1][0]
        y1 = listOf3couples[i][1][1]
        listOf3couples[i][0] = [x1, y]
        listOf3couples[i][1] = [x, y1]

    # convert from 2points representation to point+W+H
    tmp = [listOf3couples[i][0][0], listOf3couples[i][0][1],
           listOf3couples[i][1][0] - listOf3couples[i][0][0],
           listOf3couples[i][1][1] - listOf3couples[i][0][1]]

    listOf3.append(tmp)
    tmpList = []
    Img = clone.copy()
    print listOf3

bestBB = [BBB.selectBest(listOf3, imgpath)]

displayAll(bestBB, listOf3)
