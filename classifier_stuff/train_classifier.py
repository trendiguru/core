from __future__ import print_function
# !/usr/bin/env python
__author__ = 'jeremy'
import cv2
import subprocess
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from time import sleep
import cProfile, pstats, StringIO
import argparse
import logging
import sys

import Utils
import background_removal

GREEN = [0, 255, 0]
RED = [0, 0, 255]
BLUE = [255, 0, 0]

# TODO
# specific long, short, med dress classifiers
# try training on hue images or hue gradient images
# implement single_negatives_file for create_negatives_recursive
# get number of positives, negatives and dont run trainer on more than that - DONE
# add backgrounds to positive test images, see if it makes difference
# randomize order of test images (since subdirectories are ordered some way)
# redo these to all different directories for positives, negatives -  DONE
# redo this to scan all directories (and subdirs) not just numbers - DONE

def write_bbfile(fp, bb, filename, bbfilename=''):
    string = filename + ' 1 {0} {1} {2} {3} \n'.format(bb[0], bb[1], bb[2], bb[3])
    print('writing ' + str(string) + ' to ' + bbfilename)
    fp.write(string)


def read_bbfile_and_show_positives(bbfilename, parent_dir):
    try:
        with open(bbfilename, 'r') as fp:
            for line in fp:
                values = line.split()
                fname = values[0]
                fname = os.path.join(parent_dir, fname)
                bb = []
                for value in values[2:]:
                    bb.append(int(value))
                print('fname:' + fname + ' bb:' + str(bb))
                img_array = cv2.imread(fname)
                if img_array is None:
                    print('no image gotten, None returned')
                    continue
                else:
                    print('succesfully got ' + fname)
                    cv2.rectangle(img_array, (bb[0], bb[1]),
                                  (bb[0] + bb[2], bb[1] + bb[3]),
                                  GREEN, thickness=1)
                    print('bb=' + str(bb))
                    cv2.imshow('win', img_array)
                    k = cv2.waitKey(200)
                    cv2.destroyAllWindows()
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        print('oops. an environment error. take cover:' + sys.exc_info()[0])


def read_bbfile_and_show_positives_in_subdirs(parent_dir='images'):
    for dir, subdir_list, file_list in os.walk(parent_dir):
        print('Found directory: %s' % dir)
        bbfilename = os.path.join(dir, 'bbs.txt')
        read_bbfile_and_show_positives(bbfilename, dir)


def create_positives_using_faces_nonrecursive(bbfilename='bbs.txt', dir='images', item='dress', use_visual_output=False,
                                              maxfiles=10000, overwrite=False):
    maxfiles = 10
    filecounter = 0
    if overwrite:
        mode = 'w'
    else:
        mode = 'a'
    try:

        with open(bbfilename, mode) as fp:
            file_list = Utils.files_in_directory(dir)
            for file in file_list:
                print('\t%s' % file)
                img_array = cv2.imread(file)
                if not Utils.is_valid_image(img_array):
                    print('no image gotten, None returned for ' + file)
                    continue
                else:
                    print('succesfully got ' + file + ', count=' + str(filecounter))
                    bb = get_bb(img_array, use_visual_output, fname=file, item=item)
                    if bb is not None:
                        # print('bb=' + str(bb) + ' x1y1x2y2:' + str(bb[0]) + ',' + str(bb[1]) + ',' + str(
                        # bb[0] + bb[2]) + ',' + str(bb[1] + bb[3]))
                        write_bbfile(fp, bb, file)
                        filecounter = filecounter + 1
                        if filecounter > maxfiles:
                            print('filecounter exceeded 0}>{1}'.format(filecounter, maxfiles))
                            break
                            # raw_input('hit enter')
                    else:
                        print('no bb found')
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        print('oops. an environment error in nonrecursive positives generator. take cover')

#make sure that u dont wipe out dir with overwrite
def create_positives_using_faces_recursive(bbfilename='bbs.txt', parent_dir='images', item='dress', single_bbfile=True,
                                           use_visual_output=False, maxfiles=10000, overwrite=False):
    filecount = 0
    print('searching %s' % parent_dir)
    try:
        for dir, subdir_list, file_list in os.walk(parent_dir):
            print('Found directory: %s' % dir)
            if not single_bbfile:  # make a bbfile for each subdir
                bbfilename = os.path.join(dir, 'bbs.txt')
            try:
                if overwrite:
                    mode = 'w'
                else:
                    mode = 'a'
                with open(bbfilename, mode) as fp:
                    for fname in file_list:
                        #  print('found file %s' % fname)
                        full_filename = os.path.join(dir, fname)
                        # fp.write

                        img_array = cv2.imread(full_filename)
                        if not Utils.is_valid_image(img_array):
                            print(fname + ' is not a valid image')
                            continue
                            # elif not isinstance(img_array[0][0], int):
                            # print('no image gotten, not int')
                            # continue
                        else:
                            print('succesfully got ' + full_filename)
                            bb = get_bb(img_array, use_visual_output, fname=fname, item=item)
                            if bb is not None:
                                print('bb=' + str(bb) + ' x1y1x2y2:' + str(bb[0]) + ',' + str(bb[1]) + ',' + str(
                                    bb[0] + bb[2]) + ',' + str(bb[1] + bb[3]))
                                if single_bbfile:
                                    write_bbfile(fp, bb, full_filename, bbfilename=bbfilename)
                                else:
                                    write_bbfile(fp, bb, fname, bbfilename=bbfilename)
                                filecount = filecount + 1
                                if filecount > maxfiles:
                                    print('file count exceeded')
                                    Utils.safely_close(fp)
                                    return
                                    # raw_input('hit enter')
                            else:
                                print('no bb found')
                                # except:
                                # e = sys.exc_info()[0]
                                # print("could not read " + full_filename + " locally due to " + str(e) + ", returning None")
                                # logging.warning("could not read locally, returning None")
                    Utils.safely_close(fp)
            except IOError:
                logging.error('ioerror, cant find file or read data:' + bbfilename)
    except OSError:
        print('oserror, error walking directory ' + parent_dir)


def get_bb(img_array, use_visual_output=True, fname='filename', item='dress'):
    faces = background_removal.find_face(img_array, max_num_of_faces=1)
    if faces is not None and faces is not [] and faces is not () and len(faces) == 1:

        print('faces:' + str(faces))
        face = faces[0]
    else:
        return None
    if item == 'dress':
        item_length = 12
        item_width = 4
        item_y_offset = 0
    else:
        item_length = 12
        item_width = 4
        item_y_offset = 0

    orig_h, orig_w, d = img_array.shape
    head_x0 = face[0]
    head_y0 = face[1]
    w = face[2]
    h = face[3]
    item_w = w * item_width
    item_y0 = head_y0 + h + item_y_offset
    item_h = min(h * item_length, orig_h - item_y0 - 1)
    item_x0 = max(0, head_x0 + w / 2 - item_w / 2)
    item_w = min(w * item_width, orig_w - item_x0 - 1)
    item_box = [item_x0, item_y0, item_w, item_h]
    if use_visual_output == True:
        cv2.rectangle(img_array, (item_box[0], item_box[1]),
                      (item_box[0] + item_box[2], item_box[1] + item_box[3]),
                      GREEN, thickness=1)
        print('plotting img, dims:' + str(orig_w) + ' x ' + str(orig_h))
        # im = plt.imshow(img_array)
        # plt.show(block=False)

        cv2.imshow(fname, img_array)
        cv2.moveWindow('win', 100, 200)
        k = cv2.waitKey(50)
        # raw_input('enter to continue')
        cv2.destroyAllWindows()

        if k in [27, ord('Q'), ord('q')]:  # exit on ESC
            pass
    assert (Utils.bounding_box_inside_image(img_array, item_box))
    return item_box


def create_negatives_nonrecursive(dir, negatives_filename='negatives' + str(dir) + '.txt', use_visual_output=True,
                                  maxfiles=10000, overwrite=False):
    '''
        make a negatives file from a directory of images
    '''
    file_count = 0
    print('creating negatives file from ' + dir)
    file_list = Utils.files_in_directory(dir)
    if overwrite:
        fh = open(negatives_filename, "w")  # overwrite
    else:
        fh = open(negatives_filename, "a")  # append
    for file in file_list:
        print('trying file ' + file)
        if Utils.is_valid_local_image_file(file):
            if file_count > maxfiles:
                return (file_count)
            else:
                # absolute_filename = join(abs_root_dir, dirName3, file)
                img_array = cv2.imread(file)
                if img_array is not None:
                    fh.write(file + '\n')
                    file_count = file_count + 1
                    print('wrote line ' + file + ' to ' + negatives_filename + ' count=' + str(file_count))
                    if use_visual_output:
                        cv2.imshow(file, img_array)
                        #cv2.moveWindow(file, 100, 200)
                        k = cv2.waitKey(100)
                        cv2.destroyAllWindows()
                else:
                    print('image array is None')
        else:
            print('file is not vald local image')

    Utils.safely_close(fh)
    permute(negatives_filename)
    print(str(file_count) + ' negatives for directory ' + dir)
    return (file_count)


def create_negatives_recursive(dir, negatives_filename='negatives_recursive.txt', show_visual_output=True,
                               single_negatives_file=True, overwrite=False):
    '''
        make a negatives file recursively from a list of dirs
    '''

    rootDir = ''
    print('creating negatives files from ' + rootDir)
    # subprocess.call("rm -f trainingfiles/negatives*.txt", shell=True)
    #subprocess.call("rm -f negatives*.txt", shell=True)
    #subprocess.call("ls -l "+abs_root_dir, shell=True)
    #subprocess.call("pwd", shell=True)
    global top_subdirlist
    top_subdirlist = []
    abs_root_dir = rootDir

    subdirlist = [x[1] for x in os.walk(rootDir)]
    if len(subdirlist) == 0:
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


def create_negatives_from_set_of_dirs(dirlist=['images/mini', 'images/jess_good'],
                                      negatives_filename='negatives.txt', use_visual_output=True,
                                      maxfiles=20000, overwrite=False):
    if overwrite:
        fh = open(negatives_filename, "w")  # overwrite
        fh.write('')
        Utils.safely_close(fh)
    sleep(1)
    for dir in dirlist:
        create_negatives_nonrecursive(dir, negatives_filename=negatives_filename,
                                      use_visual_output=use_visual_output, maxfiles=maxfiles, overwrite=False)


def create_positives(rootDir, trainDir, infoDict, makeFrame, use_visual_output=False):
    # CREATE POSITIVE EXAMPLE FILES
    print('creating positive examples from ' + rootDir)
    global top_subdirlist
    print('top subdirs:' + str(top_subdirlist))
    subdirlist = [x[1] for x in os.walk(rootDir)]
    #    top_subdirlist = subdirlist[0]
    #print('--\nroot = ' + root)
    #splitpath=string.rsplit(root,'/',1)
    abs_root_dir = rootDir
    resultsDict = {}
    print('makeframe:' + str(makeFrame))
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
                newdir3 = dirName3 + 'framed'
                subprocess.call("mkdir " + newdir3, shell=True)

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

                        size = [2 * w, 2 * h]
                        #depth=IPL_DEPTH_8U
                        channels = 3
                        r = np.random.random_integers(low=0, high=254, size=(2 * h, 2 * w, 3))
                        #                        biggerImg = biggerImg % 2**8                 #     // convert to unsigned 8-bit

                        biggerImg = np.ones((2 * h, 2 * w, 3), np.uint8) * 255
                        #                        biggerImg=(biggerImg+r)% 2**8
                        #np.uint8)+np.rand
                        #                        biggerImg=np.random.randn(biggerImg,128,30)
                        #cv.CreateImage(size, depth, channels)
                        #biggerImg[yy  ,xx,:]=img_array[yy+r[1],xx+r[0],:]
                        #img_array.copyTo(biggerImg(Rect(w/2, h/2, img_array.cols, image_array.rows)));
                        left = w / 2
                        top = h / 2
                        right = 3 * w / 2
                        bottom = 3 * h / 2
                        #print('size:'+str(h)+'x'+str(w)+' newsize:'+str(h*2)+'x'+str(w*2)+' l,r:'+str(left)+'x'+str(right)+' t,b:'+str(top)+'x'+str(bottom))
                        biggerImg[h / 2:3 * h / 2, w / 2:3 * w / 2, :] = img_array
                        #cv2.rectangle(biggerImg,(left,top),(left+w,top+h),GREEN,thickness=1)

                        relative_filename = join(newdir3, file)
                        framedFileName = relative_filename + '.framed.jpg'
                        #print('framedfile:'+framedFileName)
                        retval = cv2.imwrite(framedFileName, biggerImg)
                        #print('retval:'+str(retval))
                        #    cv2.waitKey(100)
                        if use_visual_output is True:
                            cv2.imshow('bigger', biggerImg)
                            k = 0xFF & cv2.waitKey(1)
                        string = (
                            framedFileName + ' 1 ' + str(left) + ' ' + str(top) + ' ' + str(w) + ' ' + str(h) + '\n')
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
    # print(new_order)
    tempfilename = filename + '.tmp'
    f1 = open(filename, 'w')
    for i in range(0, n):
        f1.write(lines[new_order[i]])
    f1.flush()
    os.fsync(f1.fileno())  #this and f.flush were needed since after file close, file wasn't immediately available.
    f1.close()

def create_vecfiles(rootDir, trainDir, infoDict, inputdir, outputdir, train_width=20, train_height=20,
                    num_negatives=4000, num_positives=2000,
                    num_extra_positives=100, featureType='HAAR', mode='ALL',
                    maxFalseAlarmRate=0.4, minHitRate=0.995, precalcIdxBufSize=6000, precalcValBufSize=6000,
                    num_stages=20, delay_minutes=5):
    ##########
    # CREATESAMPLES
    ##########
    print('creating vecfile (positives)')
    subdirlist = [x[1] for x in os.walk(rootDir)]
    # top_subdirlist = subdirlist[0]
    global top_subdirlist
    print('top subdirs:' + str(top_subdirlist))
    for subdir in top_subdirlist:
        d = infoDict[subdir]
        n_pos = min(d['n_positives'], num_positives)
        print('DIR:' + subdir + ' pos:' + str(n_pos))

        bbfilename = 'positives.' + subdir + '.txt'
        vecfilename = join(trainDir, subdir)
        vecfilename = join(vecfilename, 'vecfile.' + subdir + '.vec')
        print('vecfilename:' + vecfilename)
        #    bbfilename = 'trainingfiles/bbfile.' + subdir + '.txt'
        command_string = "opencv_createsamples -info " + bbfilename + " -w " + str(train_width) + " -h " + str(
            train_height) + ' -num ' + str(n_pos) + ' -vec ' + vecfilename + ' -show'


        #	command_string="opencv_createsamples -info " +bbfilename+ " -w "+str(train_width)+" -h "+str(train_height)+' -vec trainingfiles/vecfile.'+subdir+'.vec '+'-num '+str(num_positives)
        print(command_string)
        subprocess.call(command_string, shell=True)
        #    subprocess.call(command_string, shell=True)


def new_create_vecfiles(input_filename='bbs.txt', vecfilename='classifiers_to_test/vecfile.vec', train_width=20,
                        train_height=20,
                        num_negatives=4000, num_positives=2000,
                        num_extra_positives=100, featureType='HAAR', mode='ALL',
                        maxFalseAlarmRate=0.4, minHitRate=0.995, precalcIdxBufSize=6000, precalcValBufSize=6000,
                        num_stages=20, delay_minutes=5, outputfilename='classifiers_to_test/createsamples_output.txt',
                        showinfo=True):
    ##########
    # CREATESAMPLES
    ##########
    print('creating vecfile (positives)')
    print('vecfilename:' + vecfilename)
    print('command output being written to:' + outputfilename)
    n_positives = Utils.lines_in_file(input_filename)
    print('{0} positives in {1}'.format(n_positives, input_filename))
    # bbfilename = 'trainingfiles/bbfile.' + subdir +'.txt'
    if showinfo:
        show_or_not = ' -show'
    else:
        show_or_not = ''
    command_string = "opencv_createsamples -info " + input_filename + " -w " + str(train_width) + " -h " + str(
        train_height) + ' -num ' + str(
        n_positives) + ' -vec ' + vecfilename + show_or_not + '  >> ' + outputfilename + ' 2>&1 &'

    # command_string="opencv_createsamples -info " +bbfilename+ " -w "+str(train_width)+" -h "+str(train_height)+' -vec trainingfiles/vecfile.'+subdir+'.vec '+'-num '+str(num_positives)
    print(command_string)
    subprocess.call(command_string, shell=True)
    # subprocess.call(command_string, shell=True)


def train(trainDir, train_subdir, infoDict, inputdir, outputdir, train_width=20, train_height=20, num_negatives=4000,
          num_positives=2000, num_extra_positives=100,
          maxFalseAlarmRate=0.4, minHitRate=0.995, precalcIdxBufSize=6000, precalcValBufSize=6000, mode='ALL'):
    pass


def new_train(vecfilename='vecfile.vec', negatives_filename='negatives.txt', classifier_directory='classifiers_to_test',
              train_width=20,
              train_height=20, num_negatives=4000, num_positives=2000, num_extra_positives=100,
              maxFalseAlarmRate=0.4, minHitRate=0.995, precalcIdxBufSize=6000, precalcValBufSize=6000,
              mode='ALL', num_stages=20, featureType='HAAR', start_afresh=True):
    ##########
    # TRAIN
    ##########
    # opencv_traincascade -data $DIR -vec $DIR/dresses_111014_vecfile.vec -bg dressesNegatives.txt -featureType $FEATURETYPE -w $WIDTH -h $HEIGHT -numPos 1600 -numNeg 2100 -numStages 20 -mode ALL -precalcValBufSize 30000 -precalcIdxBufSize 30000 -minHitRate 0.997 -maxFalseAlarmRate 0.5 > $DIR/trainout.txt 2>&1 &
    # -precalcValBufSize 30000 -precalcIdxBufSize 30000 -minHitRate 0.997 -maxFalseAlarmRate 0.5 > $DIR/trainout.txt 2>&1 &
    #command_string='opencv_traincascade -data trainingfiles -vec trainingfiles/vecfile.'+train_subdir+'.vec '+'-bg '+bbfilename+' -featureType '+featureType+' -w '+str(train_width)+' -h '+str(train_height)

    if start_afresh:
        Utils.purge(classifier_directory, '.xml')
    outfile = join(classifier_directory, 'trainoutput.txt')
    if not os.path.isdir(classifier_directory):
        os.makedirs(classifier_directory)

    print('positives dir ' + classifier_directory + ' neg:' + str(num_negatives) + ' pos:' + str(
        num_positives) + ' w:' + str(train_width) + ' h:' + str(
        train_height))
    # max_pos = Utils.lines_in_file()
    max_negatives = Utils.lines_in_file(negatives_filename)
    if num_negatives > max_negatives:
        logging.warning('too many negatives, {0} asked for and {1} available'.format(num_negatives, max_negatives))
    command_string = ''
    command_string = 'sudo nice -n -20 ionice -c 1 -n 0 '  # nice -n -20 has highest priorite, ionice -c 1 -n -0 has highest
    command_string = command_string + 'opencv_traincascade -data ' + classifier_directory + ' -vec ' + vecfilename + ' -bg ' + negatives_filename
    command_string = command_string + ' -featureType ' + featureType + ' -w ' + str(train_width) + ' -h ' + str(
        train_height)
    command_string = command_string + ' -numPos ' + str(int(num_positives - num_extra_positives)) + ' -numNeg ' + str(
        num_negatives) + ' -numStages ' + str(num_stages)
    command_string = command_string + ' -mode ' + mode + ' -precalcValBufSize ' + str(precalcValBufSize)
    command_string = command_string + ' -precalcIdxBufSize ' + str(precalcIdxBufSize) + ' -minHitRate ' + str(
        minHitRate)
    command_string = command_string + ' -maxFalseAlarmRate ' + str(maxFalseAlarmRate)
    command_string = command_string + ' >> ' + outfile + ' 2>&1 &'
    print(command_string)
    f1 = open(outfile, 'a')
    f1.write(command_string)
    f1.close()
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


def train_wrapper(inputdir, outputdir, train_width=20, train_height=20, num_negatives=4000, num_positives=2000,
                  num_extra_positives=100,
                  maxFalseAlarmRate=0.4, minHitRate=0.995, precalcIdxBufSize=6000, precalcValBufSize=6000, mode='ALL',
                  num_stages=20, featureType='HAAR', delay_minutes=5

                  ):
    pr = cProfile.Profile()
    pr.enable()

    global top_subdirlist

    # train_width = 20
    # train_height = 20
    # num_positives = 2000
    # num_extra_positives = int(0.1 * num_positives) + 100
    # num_negatives = 4000
    # maxFalseAlarmRate = 0.4
    # minHitRate = 0.995
    # precalcIdxBufSize = 6000
    # precalcValBufSize = 6000
    # mode = 'ALL'
    # num_stages = 20
    # featureType = 'HAAR'
    # delay_minutes = 5

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

    n_negs = create_negatives_recursive(rootDir, negatives_filename='negatives_recursive.txt', show_visual_output=True,
                                        single_negatives_file=True)

    # n_negs=[100,100,100]
    print(n_negs)
    infoDict = create_positives(rootDir, trainDir, n_negs, makeFrame=True)
    print(infoDict)
    create_vecfiles(rootDir, trainDir, infoDict)

    subdirlist = [x[1] for x in os.walk(rootDir)]
    #   top_subdirlist=subdirlist[0]
    i = 0
    for subdir in top_subdirlist:
        a = memory()
        freemem = a['free']
        print('free memory:' + str(freemem))
        while freemem < (precalcIdxBufSize + precalcValBufSize) * 1000 + 1000000:  #check if enough memory left
            sleep(delay_minutes * 60)
            a = memory()
            freemem = a['free']
            print('free memory:' + str(freemem))
        train(trainDir, subdir, infoDict)  ##############################
        i = i + 1
        sleep(60 * delay_minutes)

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


def train_wrapper2(positives_file, negatives_file, output_dir='classifier_results', train_width=20, train_height=20,
                  num_negatives=4000,
                  num_positives=2000, num_extra_positives=100, maxFalseAlarmRate=0.4, minHitRate=0.995,
                  precalcIdxBufSize=6000,
                  precalcValBufSize=6000, mode='ALL', num_stages=20, featureType='HAAR', delay_minutes=5):
    global max_files
    max_files = 100000  # this numnber of files from each other directory, max

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    create_vecfiles(output_dir, output_dir, 'hi')

    a = memory()
    freemem = a['free']
    print('free memory:' + str(freemem))
    while freemem < (precalcIdxBufSize + precalcValBufSize) * 1000 + 1000000:  # check if enough memory left
        sleep(delay_minutes * 60)
        a = memory()
        freemem = a['free']
        print('free memory:' + str(freemem))
    train(output_dir, output_dir, 'hi')  ##############################

    sleep(60 * delay_minutes)

    # s = StringIO.StringIO()
    #   sortby = 'cumulative'
    #   ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #   ps.print_stats()
    #   print(s.getvalue())

##############################
# start from here
##############################
def prepare_and_train():
    # -mode <BASIC (default) | CORE | ALL>
    # -bt <{DAB, RAB, LB, GAB(default)}>

    # box_images_and_write_bbfile_using_faces_recursive('images/dresses/bridal-dresses')
    #    box_images_and_write_bbfile(dir, use_visual_output=True)
    #    read_bbs_in_subdirs(dir)

    print('starting create_positive_and_negative_files')
    negatives_dir = 'images/womens-tops'
    negatives_dirs = ['images/womens-tops', 'images/mens-shirts']
    positives_dir = 'images/dresses'
    classifier_dir = 'classifiers_to_test/classifier109/'
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.4  # .8^20 = 0.01
    minHitRate = 0.98  # 0.995^20 = 0.9
    precalcValBufSize = 6000
    precalcIdxBufSize = 6000
    mode = 'ALL'
    num_stages = 20
    featureType = 'LBP'
    # num_pos = 2000
    #   num_neg = 5000

    print('classifier dir:' + classifier_dir)
    Utils.ensure_dir(classifier_dir)
    bb_filename = classifier_dir + 'bbs.txt'
    bb_filename = 'bbs.txt'
    negatives_filename = classifier_dir + 'negatives.txt'
    negatives_filename = 'negatives.txt'
    vecfilename = classifier_dir + 'vecfile.vec'
    # vecfilename ='vecfile.vec'
    create_samples_outputfile = classifier_dir + 'createsamplesoutput.txt'
    # create_negatives_from_set_of_dirs(dirlist=negatives_dirs, negatives_filename=negatives_filename,
    # use_visual_output=False,
    #                                  maxfiles=10000, overwrite=True)

    # create_negatives_nonrecursive(negatives_dir, negatives_filename='negatives.txt', show_visual_output=True,
    #                                 maxfiles=20000,
    #                                 overwrite=True)

    print('finished making negatives')
    # raw_input('enter to continue')
#    create_positives_using_faces_recursive(bbfilename=bb_filename, parent_dir=positives_dir,
    #item='dress', single_bbfile=True, use_visual_output=False, maxfiles=10000,
    # overwrite=False)


    new_create_vecfiles(input_filename=bb_filename, outputfilename=create_samples_outputfile, vecfilename=vecfilename,
                        showinfo=False, train_width=train_width, train_height=train_height)
    sleep(30)  # wait till vecfile write is done
    num_pos = Utils.lines_in_file(bb_filename)
    num_neg = Utils.lines_in_file(negatives_filename)
    num_extra_positives = num_pos * num_stages / 70 + 4800
    num_extra_negatives = 50

    num_pos = 2000
    num_extra_positives = 0
    num_extra_negatives = 0
    num_neg = 5000

    print('avail pos {0} avail neg {1}'.format(num_pos, num_neg))


    new_train(vecfilename=vecfilename, negatives_filename=negatives_filename,
              classifier_directory=classifier_dir, train_width=train_width,
              train_height=train_height, num_negatives=num_neg - num_extra_negatives, num_positives=num_pos,
              num_extra_positives=num_extra_positives,
              maxFalseAlarmRate=maxFalseAlarmRate, minHitRate=minHitRate, precalcIdxBufSize=precalcIdxBufSize,
              precalcValBufSize=precalcValBufSize,
              mode=mode, num_stages=num_stages, featureType=featureType, start_afresh=True)


if __name__ == "__main__":

    prepare_and_train()

    if (0):
        print('start')
        parser = argparse.ArgumentParser(description='train classifier')
        parser.add_argument('--imagedir', type=str, help='input images here',
                            default='images')  # basestring instead of str would be better but doesnt work
        parser.add_argument('--outputdir', type=str, help='output file here', default='output.txt')

        args = parser.parse_args()
        print(args)
        imagedir = args.imagedir
        outputdir = args.outputdir
        print('Input dir is ' + imagedir)
        print('output dir is ' + outputdir)
        # print 'Output file is "', outputfile

        train_wrapper(imagedir, outputdir)

'''
    # train_width = 20
    # train_height = 20
    # num_positives = 2000
    # num_extra_positives = int(0.1 * num_positives) + 100
    # num_negatives = 4000
    # maxFalseAlarmRate = 0.4
    # minHitRate = 0.995
    # precalcIdxBufSize = 6000
    # precalcValBufSize = 6000
    # mode = 'ALL'
    # num_stages = 20
    # featureType = 'HAAR'
    # delay_minutes = 5



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
sudo python train_classifier.py -i images/imageNet/easy -o 064   (longdress, shirt, suit)
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
sudo python train_classifier.py -i images/cjdb -o 065
coat  dress  hat  pants  shirt  shoe  shorts  suit  sweater
same params as 064


066
sudo python train_classifier.py -i images/cjdb -o 066
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
sudo python train_classifier.py -i images/imageNet/easy -o 067
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
sudo python train_classifier.py -i images/cjdb -o 068
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
root@ip-172-31-37-253:/home/ubuntu/fpModules/classifier_stuff# sudo python train_classifier.py -o 069 -i images/cjdb
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
sudo python train_classifier.py -o 070 -i images/cjdb
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
sudo python train_classifier.py -o 071 -i images/imageNet/easy
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
sudo python train_classifier.py -o 072 -i images/imageNet/easy
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

101   dresses positive, mens-shirts  and womens-tops negatives
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.7  # .8^20 = 0.01
    minHitRate = 0.995  # 0.995^20 = 0.9
    precalcValBufSize=6000
    precalcIdxBufSize=6000
    mode='ALL'
    num_stages=20
    featureType='HAAR'

102 had tons of false positives
102   classifier_dir = 'classifiers_to_test/classifier102/'
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.5  # .8^20 = 0.01
    minHitRate = 0.995  # 0.995^20 = 0.9
    precalcValBufSize = 6000
    precalcIdxBufSize = 6000
    mode = 'ALL'
    num_stages = 20
    featureType = 'HAAR'

103   fewer positives to not run out
    classifier_dir = 'classifiers_to_test/classifier103/'
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.3  # .8^20 = 0.01
    minHitRate = 0.98  # 0.995^20 = 0.9
    precalcValBufSize = 6000
    precalcIdxBufSize = 6000
    mode = 'ALL'
    num_stages = 20
    featureType = 'HAAR'

minhitrate 0.99 instead of 0.995
104    classifier_dir = 'classifiers_to_test/classifier104/'
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.3  # .8^20 = 0.01
    minHitRate = 0.99  # 0.995^20 = 0.9
    precalcValBufSize = 6000
    precalcIdxBufSize = 6000
    mode = 'ALL'
    num_stages = 20
    featureType = 'HAAR'

minhitrate 0.98
105    classifier_dir = 'classifiers_to_test/classifier105/'
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.3  # .8^20 = 0.01
    minHitRate = 0.98  # 0.995^20 = 0.9
    precalcValBufSize = 6000
    precalcIdxBufSize = 6000
    mode = 'ALL'
    num_stages = 20
    featureType = 'HAAR'

106 - minhit=0.97, FA=0.2
    classifier_dir = 'classifiers_to_test/classifier106/'
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.2  # .8^20 = 0.01
    minHitRate = 0.97  # 0.995^20 = 0.9
    precalcValBufSize = 6000
    precalcIdxBufSize = 6000
    mode = 'ALL'
    num_stages = 20
    featureType = 'HAAR'

107 minhit 0.96 FA 0.4 LBP
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.4  # .8^20 = 0.01
    minHitRate = 0.96  # 0.995^20 = 0.9
    precalcValBufSize = 6000
    precalcIdxBufSize = 6000
    mode = 'ALL'
    num_stages = 20
    featureType = 'LBP'

108 minhit 0.98 FA 0.4 LBP
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.4  # .8^20 = 0.01
    minHitRate = 0.96  # 0.995^20 = 0.9
    precalcValBufSize = 6000
    precalcIdxBufSize = 6000
    mode = 'ALL'
    num_stages = 20
    featureType = 'LBP'

109   2000 pos and 5000 neg, 18 stages
    classifier_dir = 'classifiers_to_test/classifier109/'
    train_width = 15
    train_height = 20
    maxFalseAlarmRate = 0.4  # .8^20 = 0.01
    minHitRate = 0.98  # 0.995^20 = 0.9
    precalcValBufSize = 6000
    precalcIdxBufSize = 6000
    mode = 'ALL'
    num_stages = 18
    featureType = 'LBP'
    num_pos = 2000
    num_neg = 5000


'''

# 337 3139
# 338 3145    15:10
# 352 3145 15:12
# 635 3151   16:12
# 891 3151 17:27

