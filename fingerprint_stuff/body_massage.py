__author__ = 'jeremy'
import os
import cv2

f = []
mypath='/home/jeremy/OpenCV/opencv-2.4.9/data/haarcascades'
for (dirpath, dirnames, filenames) in os.walk(mypath):
    f.extend(filenames)
    break

print('classifiers:'+str(f))

images = []
imgpath='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/databaseImages/googImages/Pictures for Jeremy/Pictures for Jeremy/pencil skirts - Yahoo Image Search Results_files'
for (dirpath, dirnames, imgfilenames) in os.walk(imgpath):
    images.extend(imgfilenames)
    break

#print('images:'+str(images))
cascades=[]

# maybe bug is here
for classifier in f:
    cascades.append(cv2.CascadeClassifier(classifier))
#    cascades.append(cv2.CascadeClassifier(xmlNames[0]))


for imgname in images:
        print('img:'+str(imgname))
        img=cv2.imread(os.path.join(imgpath,imgname))
        if img is not None:
            count=0
            cv2.imshow('first', img)
            k = 0xFF & cv2.waitKey(500)
            for cascade in cascades:
                count=count+1
                detectedRects = cascade.detectMultiScale(img)
                Nmatches=len(detectedRects)
                for match in detectedRects:
                    color = [color*5%180,150,150]
                    cv2.rectangle(img,(match[0],match[1]),(match[0]+match[2],match[1]+match[3]),color,2)
                    cv2.imshow('input', img)

                    k = 0xFF & cv2.waitKey(0)

    #    print('targets:'+str(Nmatches))


