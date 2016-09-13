__author__ = 'jeremy'

#!/usr/bin/env bash
import os
from trendi import Utils
import cv2
import urllib
import numpy as np
import requests
from trendi import constants
import multiprocessing
from multiprocessing import Pool
import urllib2
import subprocess
import json
import time
from pprint  import pprint
from cv2 import imdecode, imwrite


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
    i = 0
    while i < len(files):
        f = files[i]
        print('i='+str(i))
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
        print('(d)elete (a)lone (b)ack (space)nothing (q)uit ')
        c = cv2.waitKey(0)
        if c == ord('b'):
            print('go back')
            i=i-1
            continue
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
            cv2.destroyAllWindows()
            return
        else:
            print('nothing')
        n = n + 1
        i = i + 1


def save_img_at_url(url,savename=None):
    # download the image, convert it to a NumPy array, and then save
    # it into OpenCV format using last part of url (will overwrite imgs of same name at different url)
    if not savename:
        savename = url.split('?')[0]
    img_arr = Utils.get_cv2_img_array(url)
    print('name:'+savename)
    cv2.imwrite(savename,img_arr)
    return img_arr


    if url.count('jpg') > 1:
        print('np jpg here captain')
        return None
    resp = requests.get(url)
    print resp
#    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image

def getty_dl(searchphrase,n_pages = 2000,savedir=None):
    if savedir is None:
        savedir = '/home/jeremy/image_dbs/getty/'+searchphrase+'/'
    Utils.ensure_dir(savedir)
    #do a first curl to set the page size
#    cmd = 'curl -X GET -H "Api-Key: r6zm5n78dguspxkg2ss4xvje"  https://api.gettyimages.com/v3/search/images?page_size=100000 > resout1.txt'
#    res = subprocess.call(cmd,shell=True)
    #next curl with the right phrase, all subsequent ones with ?page= to get next results from same query
    outfile = searchphrase+'out.txt'
    for i in range(n_pages):
        query = '?phrase='+searchphrase+'&page='+str(i+1)
        print query
        cmd = 'curl -X GET -H "Api-Key: r6zm5n78dguspxkg2ss4xvje"  "https://api.gettyimages.com/v3/search/images'+query+ '" > ' + outfile
        print cmd
        res = subprocess.call(cmd,shell=True)
        with open(outfile,'r') as f:
            d = json.load(f)
            f.close()
            pprint(d)
        if not d:
            print('no file found')
            continue
        if not 'images' in d:
            print('no images field in result, continuing')
            continue
        imgs = d['images']
        l = len(imgs)
#        print imgs
        print l
        for j in range(l):
            time.sleep(0.05)
            nth_img = imgs[j]
  #          print nth_img
            if not 'display_sizes' in nth_img:
                print('no display sizes field found, continuing')
                continue
            ds = nth_img['display_sizes']
#            print ds
            first = ds[0]
 #           print first
            uri = first['uri']
#                print uri
            clean_url = uri.split('?')[0]
#                print(clean_url)
            savename=clean_url.split('?')[0]
            savename=savename.split('/')[-1]
            savename = searchphrase + savename
            savename = os.path.join(savedir,savename)
            Utils.ensure_dir(savedir)
#            print(savename)
            save_img_at_url(uri,savename=savename)

def getty_star(a_b):
    return getty_dl(*a_b)

if __name__=="__main__":
    items = constants.binary_cats
    items = items[3:]
#    items = [1,2,3]
#    with Pool(4) as p:
#    items = [items[0],items[1]]
#    p = Pool(len(items))
#    p.map(getty_dl, items)
#    items = ['top','sweatshirt','sweater','suit','stocking','skirt','shorts','scarf']
    items[18] = 'bikini'
    parallel = True
    if(parallel == False):
        for i in range(len(items)):
            getty_dl(items[i],n_pages=1000,savedir = '/home/jeremy/image_dbs/getty/'+items[i]+'/')
    else:
        n_proc = multiprocessing.cpu_count()
        print('nprocessors:'+str(n_proc))
        pool = multiprocessing.Pool(processes=10)
        pool.map(getty_dl, items)




