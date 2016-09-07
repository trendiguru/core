__author__ = 'jeremy'

#!/usr/bin/env bash
import os
from trendi import Utils
import cv2

def rename_goog_dirs(dir='/home/jeremy/binary_images'):
    dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]
    print('dirs:'+str(dirs))
    for d in dirs:
        print('directory:'+str(d))
        fulldir = os.path.join(dir,d)
        print('full dir '+ fulldir)
        subdirs = [d for d in os.listdir(fulldir) if os.path.isdir(os.path.join(fulldir,d))]
        print('subdirs'+str(subdirs))
        for subdir in subdirs:
            clean_dir = subdir.replace(" ","")
            clean_dir = clean_dir.replace("/","")
            clean_dir = clean_dir.replace("\\","")
            clean_dir = clean_dir.replace("-","")
            clean_dir = clean_dir.replace("GoogleSearch_files","")
            clean_dir = clean_dir.replace("'","")
            print('clean subdir:'+str(clean_dir))
            full_subdir = os.path.join(fulldir,subdir)
            print('full subdir '+str(full_subdir))
            files = [f for f in os.listdir(full_subdir) if ('jpg' in f or 'images' in f)]
            print('len files:'+str(len(files)))
            for f in files:
                destname = os.path.join(fulldir, clean_dir+'_'+f)
                if not 'jpg' in destname:
                    destname = destname+'.jpg'
                origname = os.path.join(full_subdir,f)
                print('source:'+origname+' dest:'+destname)
                os.rename(origname,destname)
    #        print files

def selectsiya(dir):
    alone_dir = os.path.join(dir,'alone')
    Utils.ensure_dir(alone_dir)
    delete_dir = os.path.join(dir,'delete_these')
    Utils.ensure_dir(delete_dir)
    files = [f for f in os.listdir(dir) if 'jpg' in f]
 #   print('files:'+str(files))
    n = 0
    for f in files:
        count_curdir = len([g for g in os.listdir(dir) if os.path.isfile(os.path.join(dir, g))])
        count_alonedir = len([g for g in os.listdir(alone_dir) if os.path.isfile(os.path.join(alone_dir, g))])
        count_deletedir = len([g for g in os.listdir(delete_dir) if os.path.isfile(os.path.join(delete_dir, g))])
        print(str(n)+' done of '+ str(count_curdir)+' files, '+str(count_alonedir)+' alone, '+str(count_deletedir)+' deleted')
        fullfile = os.path.join(dir,f)
        print('file:'+str(fullfile))
        img_arr = cv2.imread(fullfile)
        try:
            cv2.imshow('candidate',img_arr)
        except:
            print('something bad happened trying to imshow')
            destname = os.path.join(delete_dir, f)
            print('source:'+fullfile+' dest:'+destname)
            os.rename(fullfile,destname)
        c = cv2.waitKey(0)
        print('(d)elete (a)lone (space)nothing')
        if c == ord('d'):
            print('delete')
            destname = os.path.join(delete_dir, f)
            print('source:'+fullfile+' dest:'+destname)
            os.rename(fullfile,destname)
        elif c == ord('a'):
            print('alone')
            destname = os.path.join(alone_dir, f)
            print('source:'+fullfile+' dest:'+destname)
            os.rename(fullfile,destname)
        elif c == ord(' '):
            print('do nothing')
        elif c == ord('q'):
            print('quit')
            return
        else:
            print('nothing')
        n = n + 1

import urllib2
import subprocess
from subprocess import Popen, PIPE
import json
from pprint  import pprint

def getty_dl(searchphrase):

    for i in range(n_calls):
        cmd = 'curl -X GET -H "Api-Key: r6zm5n78dguspxkg2ss4xvje"  "https://api.gettyimages.com/v3/search/images?phrase='+searchphrase+'" > resout.txt'
        res = subprocess.call(cmd,shell=True)
        with open('resout.txt','r') as f:
            d = json.load(f)
            f.close()
            pprint(d)
            imgs = d['images']
            l = len(imgs)
    #        print imgs
            print l
            for i in range(l):
                nth_img = imgs[i]
      #          print nth_img
                ds = nth_img['display_sizes']
    #            print ds
                first = ds[0]
     #           print first
                uri = first['uri']
                print uri



if __name__=="__main__":
    getty_dl('shoes')