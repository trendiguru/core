__author__ = 'jeremy'
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np

    ############ make houghspace array ############
def hough_in_python():
    houghspace = []
    c = 0
    height = 400
    while c <= height:
        houghspace.append([])
        cc = 0
        while cc <= 180:
            houghspace[c].append(0)
            cc += 1
        c+=1
    ############ do transform ############
    degree_tick = 1 #by how many degrees to check
    total_votes = 1 #votes counter
    highest_vote = 0 #highest vote in the arra

    while total_votes < 100:
        img = cv2.imread('/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/Dev/trendi_guru_modules/fingerprint_stuff/testimages/images016.jpeg')
        edges = cv2.Canny(img,50,150,apertureSize = 3)
        print('angle'+str(math.pi*degree_tick/180)+' votes'+str(total_votes))
        lines = cv2.HoughLines(edges,1,math.pi*degree_tick/180,total_votes)
        try:
            for rho,theta in lines[0]:
                a = math.cos(theta)
                b = math.sin(theta)
                x1 = int((a*rho) + 1000*(-b))
                y1 = int((b*rho) + 1000*(a))
                x2 = int((a*rho) - 1000*(-b))
                y2 = int((b*rho) - 1000*(a))
                cv2.line(img,(x1,y1),(x2,y2),(50,200,255),2)
            #################add votes into the array################
            deradian = 180/math.pi #used to convert to degree
            for rho,theta in lines[0]:
                degree = int(round(theta*deradian))
                rho_pos = int(rho - 200)
                houghspace[rho_pos][degree] += 1
        #when lines[0] has no votes, it throws an error which is caught here
        except:
            total_votes = 999 #exit loop
        highest_vote = total_votes
        total_votes += 1
        del lines
    ########### loop finished ###############################
    print highest_vote
    #############################################################
    ################### plot the houghspace ###################
    maxy = 200 #used to offset the y-axis
    miny = -200 #used to offset the y-axis
    #the main grap
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_title('Houghspace')
    plt.imshow(houghspace, cmap='gist_stern')
    ax.set_aspect('equal')
    plt.yticks([0,-miny,maxy-miny], [miny,0,maxy])
    #the legend
    cax = fig.add_axes([0, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    #plot
    plt.show()

def plotlines(lines,img_array,title='lines'):
    if lines is not None:
        count=0
        for line in lines:
       #     print ('line:'+str(line))
            rho = line[0][0]
            theta = line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            multiplier = 1000
            x1 = int(x0 + multiplier*(-b))
            y1 = int(y0 + multiplier*(a))
            x2 = int(x0 - multiplier*(-b))
            y2 = int(y0 - multiplier*(a))
            cv2.line(img_array,(x1,y1),(x2,y2),(255,100,255),2)
            count=count+1
            if(count>100):
                break
        print(str(count)+ ' lines')
        cv2.imshow(title,img_array)
        k = cv2.waitKey(0) # & 0xFF

def hough_in_opencv(img_array):
    gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize=3)
    thresh=300
    lines1 = cv2.HoughLines(edges,2,np.pi/180,thresh)
    print('houghlines ok')

    if lines1 is not None:
        plotlines(lines1,edges,'theirs')
    lines2, arr = cv2.HoughLinesWithAccumulator(edges,2,np.pi/180,thresh)
    if lines2 is not None:
        plotlines(lines2,edges,'mine')#    lines2, arr = cv2.HoughLinesWithAccumulator(gray,1,np.pi/180,200)
    cv2.imshow('hough accumulator',arr)

    k = cv2.waitKey(0) # & 0xFF


#    if lines2 is not None:
#        plotlines(lines2,gray,'minegray')


import os
path="testimages"
    #os.path.join("/home","mypath","to_search")
for r,d,f in os.walk(path):
    for files in f:
       if files[-3:].lower()=='jpg' or files[-4:].lower() =="jpeg":
            filename=os.path.join(r,files)
            print "found: ",filename
            img_array=cv2.imread(filename)
            hough_in_opencv(img_array)
