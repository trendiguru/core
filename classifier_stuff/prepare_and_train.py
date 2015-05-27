from __future__ import print_function
#!/usr/bin/env python
__author__ = 'jeremy'
import cv2
import subprocess
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from time import sleep
import sys
import cProfile, pstats, StringIO
import argparse


# TODO
#get number of positives, negatives and dont run trainer on more than that - DONE
#add backgrounds to positive test images, see if it makes difference
#randomize order of test images (since subdirectories are ordered some way)
#redo these to all different directories for positives, negatives -  DONE
#redo this to scan all directories (and subdirs) not just numbers - DONE

def create_negatives(rootDir, trainDir):
    #def create_negatives(**kwargs):
    #if kwargs is not None:
    #    for key, value in kwargs.iteritems():
    #        print "%s == %s" %(key,value)

    #CREATE NEGATIVE EXAMPLE FILES - directories should be 'clean' i.e. contain only one item per image.
    #these can be used easily for both pos. and neg.  Images with two or more items can be used for negatives
    #with some more work
    print('creating negatives files from ' + rootDir)
    #subprocess.call("rm -f trainingfiles/negatives*.txt", shell=True)
    #subprocess.call("rm -f negatives*.txt", shell=True)
    #subprocess.call("ls -l "+abs_root_dir, shell=True)
    #subprocess.call("pwd", shell=True)
    global top_subdirlist
    top_subdirlist=[]

    subdirlist = [x[1] for x in os.walk(rootDir)]
    if len(subdirlist)==0:
        print('no files in directory ' + rootDir)
    exit()
    print(subdirlist)
    unpruned_top_subdirlist = subdirlist[0]
    print(unpruned_top_subdirlist)
    for a in unpruned_top_subdirlist:
        if not 'framed' in a:
            top_subdirlist.append(a)
    print(top_subdirlist)
    n_subdirs = len(top_subdirlist)
    n_negatives = {}
    for subdir1 in top_subdirlist:
        #	fh=open("trainingfiles/negatives."+subdir1+".txt","ab+")
        #		negatives_filename=join(trainDir,"negatives."+subdir1+".txt")
        negatives_filename = "negatives." + subdir1 + ".txt"
        fh = open(negatives_filename, "w")
        n_files = 0
        for subdir2 in top_subdirlist:
            if subdir1 != subdir2:
                #			print('dir1:'+subdir1+' dir2:'+subdir2)
                subdir2_path = join(rootDir, subdir2)
                #		a=raw_input()
                for dirName3, subdirList3, fileList3 in os.walk(subdir2_path, topdown=True):
                    #			mypath=join(rootDir,subdir2)
                    onlyfiles = [f for f in listdir(dirName3) if isfile(join(dirName3, f))]
                    #			print(str(len(onlyfiles))+' files in '+dirName3)
                    for file in onlyfiles:
                        #print(file)
                        if n_files > max_files:
                            break
                        else:
                            #							r1=join('..',dirName3)
                            #							relative_filename=join(r1,file)
                            relative_filename = join(dirName3, file)
                            absolute_filename = join(abs_root_dir, dirName3, file)
                            fh.write(relative_filename + '\n')
                            n_files = n_files + 1
        fh.flush()
        os.fsync(fh.fileno())  #this and f.flush were needed since after file close, file wasn't immediately available.
        fh.close()
        permute(negatives_filename)
        n_negatives[subdir1] = {'n_negatives': n_files}
        print(str(n_files) + ' negatives for directory ' + subdir1)
    return (n_negatives)

def create_positives(rootDir, trainDir, infoDict,makeFrame):
    #CREATE POSITIVE EXAMPLE FILES
    print('creating positive examples from ' + rootDir)
    global top_subdirlist
    print('top subdirs:'+str(top_subdirlist))
    subdirlist = [x[1] for x in os.walk(rootDir)]
#    top_subdirlist = subdirlist[0]
    #print('--\nroot = ' + root)
    #splitpath=string.rsplit(root,'/',1)
    resultsDict = {}
    print('makeframe:'+str(makeFrame))
    for subdir in top_subdirlist:
        #    bbfilename = rootDir+'/bbfile.' + subdir + '.txt'
        bbfilename = 'positives.' + subdir + '.txt'
        #	bbfilename = join(trainDir,'bbfile.' + subdir + '.txt')
        print('positives filename:' + bbfilename + ' subdir:' + subdir)
        cur_dir = os.path.join(rootDir, subdir)
        f = open(bbfilename, 'w')
        n_files = 0
        min_w = 10000
        min_h = 10000
        min_w_name = ''
        min_h_name = ''
        #	subdir2_path=join(rootDir,subdir)
        for dirName3, subdirList3, fileList3 in os.walk(cur_dir, topdown=True):
            #		file_path = os.path.join(rootDir,dirName3)
            #		local_dir=os.path.join(subdir,dirName3)
            #		print()
            #		print('dir3:'+dirName3+' curdir:'+cur_dir+' subdir:'+subdir)
            #		print()
            if makeFrame:
                newdir3= dirName3+'framed'
                subprocess.call("mkdir "+newdir3, shell=True)

            for file in fileList3:
                full_name = os.path.join(dirName3, file)
                #		print('fullName:'+full_name)
                img_array = cv2.imread(full_name)
                if img_array is not None:
                    relative_filename = join(dirName3, file)
                    absolute_filename = join(abs_root_dir, file)
                    h, w, d = img_array.shape
                    if makeFrame is False:
                        string = (relative_filename + ' 1 1 1 ' + str(w - 2) + ' ' + str(h - 2) + '\n')
                        #					string=(absolute_filename + ' 1 1 1 '+ str(w-2) + ' ' + str(h-2)  + '\n')
                        f.write(string)
                        #					print('file:' + full_name + ' shape:' + str(w) +'X'+ str(h))  #            			with open(file_path, 'rb') as f:
                        #					print('writing string:'+string)
                    else:  #make a frame
#                        absolute_filename = join(abs_root_dir, file)

                        size=[2*w,2*h]
                        #depth=IPL_DEPTH_8U
                        channels=3
                        r=np.random.random_integers(low=0,high=254, size=(2*h,2*w,3))
#                        biggerImg = biggerImg % 2**8                 #     // convert to unsigned 8-bit

                        biggerImg=np.ones((2*h,2*w,3),np.uint8)*255
#                        biggerImg=(biggerImg+r)% 2**8
                        #np.uint8)+np.rand
#                        biggerImg=np.random.randn(biggerImg,128,30)
                            #cv.CreateImage(size, depth, channels)
                        #biggerImg[yy  ,xx,:]=img_array[yy+r[1],xx+r[0],:]
                        #img_array.copyTo(biggerImg(Rect(w/2, h/2, img_array.cols, image_array.rows)));
                        left=w/2
                        top=h/2
                        right=3*w/2
                        bottom=3*h/2
                        #print('size:'+str(h)+'x'+str(w)+' newsize:'+str(h*2)+'x'+str(w*2)+' l,r:'+str(left)+'x'+str(right)+' t,b:'+str(top)+'x'+str(bottom))
                        biggerImg[h/2:3*h/2, w/2:3*w/2,:] = img_array
                        #cv2.rectangle(biggerImg,(left,top),(left+w,top+h),GREEN,thickness=1)

                        relative_filename = join(newdir3, file)
                        framedFileName=relative_filename+'.framed.jpg'
                        #print('framedfile:'+framedFileName)
                        retval=cv2.imwrite(framedFileName, biggerImg)
                        #print('retval:'+str(retval))
                        #    cv2.waitKey(100)
                        if use_visual_output is True:
                            cv2.imshow('bigger', biggerImg)
                            k = 0xFF & cv2.waitKey(1)
                        string = (framedFileName + ' 1 '+str(left)+' '+str(top)+ ' ' + str(w) + ' ' + str(h) + '\n')
                        #					string=(absolute_filename + ' 1 1 1 '+ str(w-2) + ' ' + str(h-2)  + '\n')
                        f.write(string)

                    n_files = n_files + 1

                    if h < min_h:
                        min_h_name = full_name
                        min_h = h
                    if w < min_w:
                        min_w_name = full_name
                        min_w = w
                    if use_visual_output is True:
                        cv2.imshow('input', img_array)
                        #    cv2.waitKey(100)
                        k = 0xFF & cv2.waitKey(1)
                else:
                    print('file:' + full_name + ' read error')  #

        f.flush()
        os.fsync(f.fileno())  #this and f.flush were needed since after file close, file wasn't immediately available.
        f.close()
        permute(bbfilename)

        print(str(n_files) + ' positive files for directory ' + str(subdir))
        #print('min height:'+str(min_h)+' in file:'+min_h_name)
        #print('min width:'+str(min_w)+' in file:'+min_w_name)
        #resultsDict[subdir]={'n_positives':n_files, 'min_h':min_h, 'min_w':min_w}
        a = infoDict[subdir]
        n_negs = a['n_negatives']
        resultsDict[subdir] = {'n_negatives': n_negs, 'n_positives': n_files, 'min_h': min_h, 'min_w': min_w}
    return resultsDict

def permute(filename):
    f1 = open(filename)
    lines = f1.readlines()
    f1.close()
    n = len(lines)
    print('shuffling file, ' + str(n) + ' lines')
    new_order = np.random.permutation(n)
    #print(new_order)
    tempfilename = filename + '.tmp'
    f1 = open(filename, 'w')
    for i in range(0, n):
        f1.write(lines[new_order[i]])
    f1.flush()
    os.fsync(f1.fileno())  #this and f.flush were needed since after file close, file wasn't immediately available.
    f1.close()

def create_vecfiles(rootDir, trainDir, infoDict):
    ##########
    # CREATESAMPLES
    ##########
    print('creating vecfile (positives)')
    subdirlist = [x[1] for x in os.walk(rootDir)]
 #   top_subdirlist = subdirlist[0]
    global top_subdirlist
    print('top subdirs:'+str(top_subdirlist))
    for subdir in top_subdirlist:
        d = infoDict[subdir]
        n_pos = min(d['n_positives'], num_positives)
        print('DIR:' + subdir + ' pos:' + str(n_pos))

        bbfilename = 'positives.' + subdir + '.txt'
        vecfilename = join(trainDir, subdir)
        vecfilename = join(vecfilename, 'vecfile.' + subdir + '.vec')
        print('vecfilename:' + vecfilename)
        #		bbfilename = join(trainDir,'bbfile.' + subdir + '.txt')
        #		vecfilename= join(trainDir,'vecfile.' + subdir + '.txt')
        #    bbfilename = rootDir+'/bbfile.' + subdir + '.txt'
        #    bbfilename = os.path.join(rootDir,'bbfile.' + subdir + '.txt')
        #    bbfilename = 'trainingfiles/bbfile.' + subdir + '.txt'
        command_string = "opencv_createsamples -info " + bbfilename + " -w " + str(train_width) + " -h " + str(train_height) + ' -num ' + str(n_pos) + ' -vec ' + vecfilename

        #	command_string="opencv_createsamples -info " +bbfilename+ " -w "+str(train_width)+" -h "+str(train_height)+' -vec trainingfiles/vecfile.'+subdir+'.vec '+'-num '+str(num_positives)
        print(command_string)
        subprocess.call(command_string, shell=True)
        #    subprocess.call(command_string, shell=True)

def train(trainDir, train_subdir, infoDict):
    ##########
    # TRAIN
    ##########
    #opencv_traincascade -data $DIR -vec $DIR/dresses_111014_vecfile.vec -bg dressesNegatives.txt -featureType $FEATURETYPE -w $WIDTH -h $HEIGHT -numPos 1600 -numNeg 2100 -numStages 20 -mode ALL -precalcValBufSize 30000 -precalcIdxBufSize 30000 -minHitRate 0.997 -maxFalseAlarmRate 0.5 > $DIR/trainout.txt 2>&1 &
    # -precalcValBufSize 30000 -precalcIdxBufSize 30000 -minHitRate 0.997 -maxFalseAlarmRate 0.5 > $DIR/trainout.txt 2>&1 &
    #command_string='opencv_traincascade -data trainingfiles -vec trainingfiles/vecfile.'+train_subdir+'.vec '+'-bg '+bbfilename+' -featureType '+featureType+' -w '+str(train_width)+' -h '+str(train_height)

    bbfilename = 'positives.' + train_subdir + '.txt'
    #	bbfilename = join(trainDir,'bbfile.' + train_subdir + '.txt')
    vecfilename = 'vecfile.' + train_subdir + '.vec '
    vecfilename = join(trainDir, train_subdir)
    vecfilename = join(vecfilename, 'vecfile.' + train_subdir + '.vec')
    #	vecfilename = join(trainDir,'vecfile.'+train_subdir+'.vec ')
    negatives_filename = "negatives." + train_subdir + ".txt"
    #	negatives_filename=join(trainDir,"negatives."+train_subdir+".txt")
    data_directory = join(trainDir, train_subdir)
    data_directory = trainDir + '/' + train_subdir
    outfile = join(data_directory, 'output.txt')
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    train_w = train_width
    train_h = train_height
    n_pos = num_positives
    d = infoDict[train_subdir]
    n_neg = min(d['n_negatives'], num_negatives)
    n_pos = min(d['n_positives'], num_positives)
    train_w = min(d['min_w'], train_width)
    train_h = min(d['min_h'], train_height)
    print('dir ' + train_subdir + ' neg:' + str(n_neg) + ' pos:' + str(n_pos) + ' w:' + str(train_w) + ' h:' + str(
        train_h))

    print('data dir:' + data_directory + ' outfile:' + outfile)
    command_string = 'nice -n19 ionice -c 3 '
    command_string=''
    command_string = command_string + 'opencv_traincascade -data ' + data_directory + ' -vec ' + vecfilename + ' -bg ' + negatives_filename
    command_string = command_string +  ' -featureType ' + featureType + ' -w ' + str(train_w) + ' -h ' + str(train_h)
    command_string = command_string  + ' -numPos ' + str(int(n_pos - num_extra_positives)) + ' -numNeg ' + str(
        n_neg) + ' -numStages ' + str(num_stages)
    command_string = command_string + ' -mode ' + mode + ' -precalcValBufSize ' + str(precalcValBufSize)
    command_string = command_string + ' -precalcIdxBufSize ' + str(precalcIdxBufSize) + ' -minHitRate ' + str(
        minHitRate)
    command_string = command_string +  ' -maxFalseAlarmRate ' + str(maxFalseAlarmRate)
    command_string = command_string + ' > ' + outfile + ' 2>&1 &'
    print(command_string)
#    subprocess.call('uptime', shell=True)

#    p=subprocess.Popen(['/bin/bash','nice','-n19'])
#    p=subprocess.Popen(['/bin/bash','nice','-n19'])
    subprocess.call(command_string, shell=True)

def memory():
    """
    Get node total memory and memory usage
    """
    with open('/proc/meminfo', 'r') as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                ret['total'] = int(sline[1])
            elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                tmp += int(sline[1])
        ret['free'] = tmp
        ret['used'] = int(ret['total']) - int(ret['free'])
    return ret


def wrapper(argv):

   pr = cProfile.Profile()
   pr.enable()
   global train_width
   global train_height
   global num_negatives
   global num_positives
   global num_extra_positives
   global featureType
   global num_stages
   global mode
   global precalcValBufSize
   global precalcIdxBufSize
   global minHitRate
   global maxFalseAlarmRate

   global top_subdirlist

   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.4
   minHitRate = 0.995
   precalcIdxBufSize = 6000
   precalcValBufSize = 6000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 2000
   num_extra_positives = int(0.1*num_positives)+100
   num_negatives =4000
   delay_minutes=5

#   print('psutil')
#   print(psutil.cpu_percent())
   #prepare_and_train()
   global use_visual_output
   use_visual_output = False
   global abs_root_dir
   abs_root_dir = '/home/www-data/web2py/applications/fingerPrint/modules/classifier_stuff/images/fashion-data/images'
   #abs_root_dir = '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/Backend/code/classifier_stuff'
   global max_files
   max_files = 100000  #this numnber of files from each other directory, max

   trainDir = outputdir
   #rootDir = 'images/fashion-data/small'
   rootDir = 'images/fashion-data/images'
   rootDir = imagedir
   #	rootDir = '../../../databaseImages/temp'

   if not os.path.isdir(trainDir):
       os.makedirs(trainDir)

   #    permute('test.txt')

   n_negs=create_negatives(rootDir,trainDir)
   # n_negs=[100,100,100]
   print(n_negs)
   infoDict=create_positives(rootDir,trainDir,n_negs,makeFrame=True)
   print(infoDict)
   create_vecfiles(rootDir,trainDir,infoDict)

   subdirlist=[x[1] for x in os.walk(rootDir)]
#   top_subdirlist=subdirlist[0]
   i=0
   for subdir in top_subdirlist:
       a=memory()
       freemem=a['free']
       print('free memory:'+str(freemem))
       while freemem<(precalcIdxBufSize+precalcValBufSize)*1000+1000000:  #check if enough memory left
           sleep(delay_minutes*60)
           a=memory()
           freemem=a['free']
           print('free memory:'+str(freemem))
       train(trainDir, subdir, infoDict)  ##############################
       i=i+1
       sleep(60*delay_minutes)

   pr.disable()
   s = StringIO.StringIO()
   sortby = 'cumulative'
   ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
   ps.print_stats()
   print(s.getvalue())

##############################
#start from here
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train classifier')
    parser.add_argument('--imagedir', type=basestring, help='directory w examples')
    parser.add_argument('--outputdir', type=basestring, help='output file', default='output.txt')

    args = parser.parse_args()
    print(args)

    print('start')
    imagedir = '057'
    outputdir = 'images/cjdb'

    print('Input dir is ' + imagedir)
    print('output dir is ' + outputdir)
    # print 'Output file is "', outputfile




    wrapper(sys.argv[1:])

'''
52
    global train_width
    train_width=20
    global train_height
    train_height=20
    global num_negatives
    num_negatives=1000
    global num_positives
    num_positives=200
    global featureType
    featureType='HAAR'
    global num_stages
    num_stages=20
    global mode
    mode='ALL'
    global precalcValBufSize
    precalcValBufSize=10000
    global precalcIdxBufSize
    precalcIdxBufSize=10000
    global minHitRate
    minHitRate=0.999
    global maxFalseAlarmRate
    maxFalseAlarmRate=0.5

51
    num_negatives=2000
    global num_positives
    num_positives=2000

55
    train_width=20
    train_height=20
    maxFalseAlarmRate=0.5
    minHitRate=0.999
    precalcIdxBufSize=10000
    precalcValBufSize=10000
    mode='ALL'
    num_stages=20
    featureType='HAAR'
    num_extra_positives=500
    num_positives=2500
    num_negatives=3000

    56
    train_width=20
    train_height=20
    maxFalseAlarmRate=0.5
    minHitRate=0.999
    precalcIdxBufSize=10000
    precalcValBufSize=10000
    mode='ALL'
    num_stages=20
    featureType='HAAR'
    num_extra_positives=500
    num_positives=4000
    num_negatives=4000

    all to here had cropped positives (3/4 pictures).
    from 57 on full positives

    057
    train_width=20
    train_height=20
    maxFalseAlarmRate=0.5
    minHitRate=0.999
    precalcIdxBufSize=10000
    precalcValBufSize=10000
    mode='ALL'
    num_stages=20
    featureType='HAAR'
    num_extra_positives=500
    num_positives=4000
    num_negatives=4000

   058
    num_positives=7000
    num_negatives=7000
    minHitRate = 0.998
    precalcIdxBufSize = 5000
    precalcValBufSize = 5000
    num_extra_positives=0.1*num_positives

   058
    maxFalseAlarmRate = 0.5
    minHitRate = 0.998
    precalcIdxBufSize = 5000
    precalcValBufSize = 5000
    mode = 'ALL'
    num_stages = 20
    featureType = 'HAAR'
    num_positives = 7000
    num_extra_positives = 0.1*num_positives
    num_negatives = 7000

   059
    maxFalseAlarmRate = 0.5
    minHitRate = 0.995
    precalcIdxBufSize = 5000
    precalcValBufSize = 5000
    mode = 'ALL'
    num_stages = 20
    featureType = 'HAAR'
    num_positives = 10000
    num_extra_positives = 0.1*num_positives
    num_negatives = 10000


060
   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.5
   minHitRate = 0.995
   precalcIdxBufSize = 2000
   precalcValBufSize = 2000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 5000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =9000
   delay_minutes=5

061
   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.8
   minHitRate = 0.995
   precalcIdxBufSize = 2000
   precalcValBufSize = 2000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 5000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =9000
   delay_minutes=5

062 

   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.7
   minHitRate = 0.995
   precalcIdxBufSize = 2000
   precalcValBufSize = 2000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 7000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =11000
   delay_minutes=5

063 using cjdb

   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.7
   minHitRate = 0.995
   precalcIdxBufSize = 2000
   precalcValBufSize = 2000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 7000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =11000
   delay_minutes=5

064 using cjdb
sudo python prepare_and_train.py -i images/imageNet/easy -o 064   (longdress, shirt, suit)
  train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.6
   minHitRate = 0.995
   precalcIdxBufSize = 2000
   precalcValBufSize = 2000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 8000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =50000
   delay_minutes=5

065
sudo python prepare_and_train.py -i images/cjdb -o 065
coat  dress  hat  pants  shirt  shoe  shorts  suit  sweater
same params as 064


066
sudo python prepare_and_train.py -i images/cjdb -o 066
coat  dress  hat  pants  shirt  shoe  shorts  suit
   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.5
   minHitRate = 0.995
   precalcIdxBufSize = 2000
   precalcValBufSize = 2000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 8000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =50000
   delay_minutes=5

067
sudo python prepare_and_train.py -i images/imageNet/easy -o 067
longDress  shirt  suit


   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.4
   minHitRate = 0.998
   precalcIdxBufSize = 15000
   precalcValBufSize = 15000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 8000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =12000
   delay_minutes=5


068
sudo python prepare_and_train.py -i images/cjdb -o 068
'shorts', 'dress', 'suit', 'shirt', 'shoe', 'pants'
   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.5
   minHitRate = 0.995
   precalcIdxBufSize = 2000
   precalcValBufSize = 2000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 8000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =50000
   delay_minutes=5

069
root@ip-172-31-37-253:/home/ubuntu/fpModules/classifier_stuff# sudo python prepare_and_train.py -o 069 -i images/cjdb
['dress', 'shirt', 'shoe', 'pants']
   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.4
   minHitRate = 0.998
   precalcIdxBufSize = 8000
   precalcValBufSize = 8000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 8000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =12000
   delay_minutes=5

070
sudo python prepare_and_train.py -o 070 -i images/cjdb
['dress', 'shirt', 'shoe', 'pants']
   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.4
   minHitRate = 0.998
   precalcIdxBufSize = 8000
   precalcValBufSize = 8000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 4000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =8000
   delay_minutes=5
didn't fnish training - took low memory and 4 days 

071 
sudo python prepare_and_train.py -o 071 -i images/imageNet/easy
['longDress', 'shirt', 'suit',]

   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.4
   minHitRate = 0.995
   precalcIdxBufSize = 6000
   precalcValBufSize = 6000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 2000
   num_extra_positives = int(0.1*num_positives)
   num_negatives =4000
   delay_minutes=5
not enough positivies so adding 100 to num_extra_positives


072
sudo python prepare_and_train.py -o 072 -i images/imageNet/easy
['longDress', 'shirt', 'suit',]

   train_width = 20
   train_height = 20
   maxFalseAlarmRate = 0.4
   minHitRate = 0.995
   precalcIdxBufSize = 6000
   precalcValBufSize = 6000
   mode = 'ALL'
   num_stages = 20
   featureType = 'HAAR'
   num_positives = 2000
   num_extra_positives = int(0.1*num_positives) +100
   num_negatives =4000
   delay_minutes=5


'''