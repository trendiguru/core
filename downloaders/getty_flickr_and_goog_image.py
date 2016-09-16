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
    start_time=time.time()
    while i < len(files):
        f = files[i]
        print('i='+str(i))
        count_curdir = len([g for g in os.listdir(dir) if os.path.isfile(os.path.join(dir, g))])
        count_alonedir = len([g for g in os.listdir(alone_dir) if os.path.isfile(os.path.join(alone_dir, g))])
        count_deletedir = len([g for g in os.listdir(delete_dir) if os.path.isfile(os.path.join(delete_dir, g))])
        print(str(n)+' done of '+ str(count_curdir)+' files, '+str(count_alonedir)+' alone, '+str(count_deletedir)+' deleted, tpi='+str((time.time()-start_time)/(i+1)))
        fullfile = os.path.join(dir,f)
        print('file:'+str(fullfile))
        try:
            img_arr = cv2.imread(fullfile)
        except:
            print('something bad happened trying to imread')
            continue
        try:
            h,w = img_arr.shape[0:2]
            img_arr = cv2.resize(img_arr,dsize=(w*2,h*2))
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

def getty_dl(searchphrase,avoid_these_terms=None,n_pages = 20000,savedir=None):
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
        skip_this = False
        for j in range(l):
            time.sleep(0.05)
            nth_img = imgs[j]
            if avoid_these_terms:
                skip_this = False
                #go thru the entire dict and check if terms to avoid is in there somewhere
                for k,v in nth_img.iteritems():
                    for item in avoid_these_terms:
#                        print('item:'+item+' k,v:'+str(k)+':'+str(v)+' type:'+str(type(v)))
                        if v and (type(v) is str or type(v) is unicode) and item in v.lower():
                            skip_this = True
                            print('SKIPPING due to :'+str(k)+':'+str(v))
                            break
                    if skip_this:
                        break
         #       raw_input('ret to cont')
  #          print nth_img
            if skip_this:
                continue
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

def flickr_get_dates(tag,mintime=0,savedir=None,n_pages=9):
    time.sleep(2)

    time_inc = 3600*24*1  #1 day
    print('getting dates, min is '+str(mintime))
    if savedir is None:
        savedir = '/home/jeremy/image_dbs/flickr/'+tag+'/'
    Utils.ensure_dir(savedir)
    compressed_tag = tag.replace(' ','+')
    outfile = compressed_tag+'out.txt'
    print('outfile:'+outfile)
    maxtime = mintime+time_inc
    pages=0
    oldpages = -1
    while(pages<n_pages and  maxtime<time.time()):
        time.sleep(3)
        initial_query = '&tags='+tag+'&min_upload_date='+str(mintime)+'&max_upload_date='+str(maxtime)+'&per_page=500'
        initial_query = '&text='+compressed_tag+'&min_upload_date='+str(mintime)+'&max_upload_date='+str(maxtime)+'&per_page=500'
        print('trying dates '+str(mintime)+' and '+str(maxtime))
        cmd = 'curl -X GET "https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key=d8548143cce923734f4093b4b063bc4f&format=json'+initial_query+'" > ' + outfile
        res = subprocess.call(cmd,shell=True)

        with open(outfile,'r') as f:
            content=f.read()
            #the info is returned inside a function call like flickerApi({bla:bla...}) so I need to strip the beginning and end stuff
            stripped = content[14:-1]
            f.close()
        try:
            d = json.loads(stripped)
        except:
            print('get_dates json problem')
            print('problem josn:'+str(stripped))
            maxtime = maxtime + time_inc
            continue
    #    pprint(d)
        if not d:
            print('no file found')
        if not 'photos' in d:
            print('no photos field in result, continuing')
        phot = d['photos']
        if not 'photo' in phot:
            print('no photo field in result, continuing')
        if 'page' in phot:
            print('page '+str(phot['page']))
            page = phot['page']
        if 'pages' in phot:
            print('of total pages '+str(phot['pages']))
            pages = phot['pages']
            if pages==oldpages:
                time_inc = time_inc*1.2
        print('got '+str(pages)+' pages')
        maxtime = int(maxtime + time_inc)
        oldpages=pages
    maxtime = maxtime - time_inc
    print('returning maxtime:'+str(maxtime)+' pages:'+str(pages))
    return maxtime


def flickr_dl(tag,avoid_these_terms=None,n_pages = 20000,start_page=1,savedir=None):
    '''
    https://www.flickr.com/services/api/flickr.photos.search.html  tags (Optional)
                A comma-delimited list of tags. Photos with one or more of the tags listed will be returned. You can exclude results that match a term by prepending it with a - character.
    :param tag:
    :param avoid_these_terms:
    :param n_pages:
    :param start_page:
    :param savedir:
    :return:
    '''
    results_per_page = 500
    max_pages_returned = int(float(4900)/results_per_page)
    compressed_tag = tag.replace(' ','+')
    if savedir is None:
        savedir = '/home/jeremy/image_dbs/flickr/'+compressed_tag+'/'
    Utils.ensure_dir(savedir)
    outfile = compressed_tag+'out.txt'
    n_dl = 0
    mintime = 1262307661  #jan 2010
    n_dates = n_pages/max_pages_returned
    for dateloop in range(n_dates):
        time.sleep(1)

        maxtime = flickr_get_dates(tag,mintime,savedir=savedir,n_pages=max_pages_returned)
        print('mintime '+str(mintime)+' maxtime:'+str(maxtime))

        for i in range(start_page,start_page+n_pages):
            query = '&tags='+tag+'&page='+str(i)+'&min_upload_date='+str(mintime)+'&max_upload_date='+str(maxtime)+'&per_page='+str(results_per_page)
            query = '&text='+compressed_tag+'&page='+str(i)+'&min_upload_date='+str(mintime)+'&max_upload_date='+str(maxtime)+'&per_page='+str(results_per_page)
            print query
            #kyle key 6351acc69daa0868c61319df617780c0   secret b7a74cf16401856b
            cmd = 'curl -X GET "https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key=d8548143cce923734f4093b4b063bc4f&format=json'+query+'" > ' + outfile
            print cmd
            res = subprocess.call(cmd,shell=True)
            print('res:'+str(type(res))+':'+str(res))
            with open(outfile,'r') as f:
                content=f.read()
                #the info is returned inside a function call like flickerApi({bla:bla...}) so I need to strip the beginning and end stuff
                stripped = content[14:-1]
                f.close()
            try:
                d = json.loads(stripped)
            except:
                print('json problem '+str(stripped))
                continue
            pprint(d)
            if not d:
                print('no file found')
                continue
            if not 'photos' in d:
                print('no photos field in result, continuing')
                continue
            phot = d['photos']
            if not 'photo' in phot:
                print('no photo field in result, continuing')
                continue
            if 'page' in phot:
                print('page '+str(phot['page']))
                page = phot['page']
            if 'pages' in phot:
                print('of total pages '+str(phot['pages']))
                pages = phot['pages']
            if page and pages and page>pages:
                print('beyond last page')
                return
            imgs = phot['photo']
            l = len(imgs)
    #        print imgs
            print(str(l)+' images found')
            skip_this = False

            for j in range(l):
 #               time.sleep(0.05)
                nth_img = imgs[j]
                if avoid_these_terms:
                    #go thru the entire dict and check if terms to avoid is in there somewhere
                    for k,v in nth_img.iteritems():
                        for item in avoid_these_terms:
    #                        print('item:'+item+' k,v:'+str(k)+':'+str(v)+' type:'+str(type(v)))
                            if v and (type(v) is str or type(v) is unicode) and item in v.lower():
                                skip_this = True
                                print('SKIPPING due to :'+str(k)+':'+str(v))
                                break
                        if skip_this:
                            break
             #       raw_input('ret to cont')
      #          print nth_img
                if skip_this:
                    continue
                if not 'farm' in nth_img:
                    print('no farm found, continuing')
                    continue
                farm = nth_img['farm']
                if not 'id' in nth_img:
                    print('no id found, continuing')
                    continue
                id = nth_img['id']
                if not 'server' in nth_img:
                    print('no server found, continuing')
                    continue
                server = nth_img['server']
                if not 'secret' in nth_img:
                    print('no secret found, continuing')
                    continue #
                secret = nth_img['secret']
                url = 'https://farm'+str(farm)+'.staticflickr.com/'+str(server)+'/'+str(id)+'_'+str(secret)+'.jpg'
      #          print url
                savename=str(id)+'.'+str(server)+'.'+str(farm)+'.jpg'
                savename = compressed_tag + savename
                savename = os.path.join(savedir,savename)
                Utils.ensure_dir(savedir)
                if os.path.exists(savename):
                    print(savename+' exists!!')
                    continue
#                    savename=savename[:-4]+'.b.jpg'
    #                raw_intput('check the flies')
                save_img_at_url(url,savename=savename)
                n_dl = n_dl + 1
            n_files = len(os.listdir(savedir))
            print('n dl:'+str(n_dl)+' n_files:'+str(n_files)+' '+savename)
        mintime = maxtime

if __name__=="__main__":
    items = constants.binary_cats
    items = items[27:]
    print items
    raw_input('ret to cont')
#    items = [1,2,3]
#    with Pool(4) as p:
#    items = [items[0],items[1]]
#    p = Pool(len(items))
#    p.map(getty_dl, items)
#    items = ['top','sweatshirt','sweater','suit','stocking','skirt','shorts','scarf']
   # items[18] = 'bikini'
    parallel = True
    if(parallel == False):
        for i in range(len(items)):
#            getty_dl(items[i],n_pages=1000,savedir = '/home/jeremy/image_dbs/getty/'+items[i]+'/')
            flickr_dl(items[i],n_pages=2000,savedir = '/home/jeremy/image_dbs/flickr/'+items[i]+'/')
    else:
        n_proc = multiprocessing.cpu_count()
        print('nprocessors:'+str(n_proc))
        pool = multiprocessing.Pool(processes=10)
#        pool.map(getty_dl, items)
        pool.map(flickr_dl, items)




