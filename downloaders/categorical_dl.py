from __future__ import print_function

__author__ = 'jeremy'
import os
import logging
import time

import cv2
from rq import Queue
from operator import itemgetter
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import time
import pymongo
import multiprocessing


from trendi import constants
from trendi.constants import db
from trendi.constants import redis_conn
import trendi.Utils as Utils
import trendi.background_removal as background_removal
from trendi.find_similar_mongo import get_all_subcategories
from trendi.utils import imutils

# download_images_q = Queue('download_images', connection=redis_conn)  # no args implies the default queue
logging.basicConfig(level=logging.WARNING)
# LESSONS: CANNOT PUT MULTIPLE PHRASES IN $text


def get_db_fields(collection='products'):
    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    cursor = db[collection].find()
    print('returned cursor')
    if cursor is None:  # make sure training collection exists
        print('couldnt get cursor ' + str(collection))
        return {"success": 0, "error": "could not get collection"}
    doc = next(cursor, None)
    i = 0
    n = cursor.count()
    print('found '+str(n)+' items in db '+collection)
    while i < n:
        print('checking doc #' + str(i + 1)+' of '+str(n))
        for k,v in doc.iteritems():
            try:
                print('key:' + str(k))
                print('value:'+str(v))
            except UnicodeEncodeError:
                print('unicode encode error')
        i = i + 1
        doc = next(cursor, None)
        print('')
        raw_input('enter key for next doc')
    return {"success": 1}



def step_thru_db(collection='products'):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    cursor = db.products.find()
    print('returned cursor')
    if cursor is None:  # make sure training collection exists
        print('couldnt get cursor ' + str(collection))
        return {"success": 0, "error": "could not get colelction"}
    doc = next(cursor, None)
    i = 0
    while doc is not None:
        print('checking doc #' + str(i + 1))
        if 'categories' in doc:
            try:
                print('cats:' + str(doc['categories']))
            except UnicodeEncodeError:
                print('unicode encode error in description')
                s = doc['categories']
                print(s.encode('utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        if 'description' in doc:
            try:
                print('desc:' + str(doc['description']))
            except UnicodeEncodeError:
                print('unicode encode error in description')
                s = doc['description']
                print(s.encode('utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        i = i + 1
        doc = next(cursor, None)
        print('')
        raw_input('enter key for next doc')
    return {"success": 1}

def find_products_by_description_and_category(search_string, category_id):
    logging.info('****** Starting to find {0} in category {1} *****'.format(search_string,category_id))

    query = {"$and": [{"$text": {"$search": search_string}},
                      {"categories":
                           {"$elemMatch":
                                {"id": {"$in": get_all_subcategories(db.categories, category_id)}
                                 }
                            }
                       }]
             }
    fields = {"categories": 1, "id": 1, "description": 1}
    cursor = db.products.find(query, fields).batch_size(10)
    logging.info("Found {count} products in cat {category} with string {search_string}".format(count=cursor.count(),
                                                                    category=category_id,
                                                                    search_string=search_string))
    return cursor

def find_products_by_category(category_id):
    logging.info('****** Starting to find category {} *****'.format(category_id))

    query = {"categories":
                           {"$elemMatch":
                                {"id": {"$in": get_all_subcategories(db.categories, category_id)}
                                 }
                            }
             }
    print('query is:'+str(query))
    fields = {"categories": 1, "id": 1, "description": 1}
    cursor = db.products.find(query, fields).batch_size(10)
    logging.info("Found {count} products in cat {category} ".format(count=cursor.count(),
                                                                    category=category_id))
    return cursor

def exhaustive_search(dl=True,dl_dir='./',use_visual_output=False,resize=(256,256),in_docker=True,parallel=True,exclude=['yonatan','temp'],min_items=1000):
    '''
    if you set the environment vars right then no need for in_docker
    :param dl:
    :param dl_dir:
    :param use_visual_output:
    :param resize:
    :param in_docker:
    :return:
    '''
    if in_docker:
        db = pymongo.MongoClient('localhost',port=27017).mydb
    else:
        db = constants.db
    collections = filter(lambda coll: not any([term in coll.lower() for term in exclude]), db.collection_names())
    for collection in collections:
        dl_collection(collection,db,parallel=parallel)

def dl_collection(collection,db,parallel=True):
    print('checking collection '+str(collection))
#        cursor = db.collection.find() #wont work since collceiton is a string
    cursor = db[collection].find()
    count = cursor.count()
    if count == 0:
        print('no items....')
        return
    print('collection {} has {} items'.format(collection,count))
    cats = db[collection].distinct('categories')
    print('categories: '+str(cats))
    jobs=[]
    for cat in cats:
        time.sleep(0.1)
        if parallel:
#            p = multiprocessing.Pool(30)
#               p.map(simple_cat_dl, args=(cats,vargs)
            p = multiprocessing.Process(target=simple_cat_dl,args=(cat,collection,db,dl,dl_dir,use_visual_output,resize))
            jobs.append(p)
            p.start()
        else:
            simple_cat_dl(cat,collection,db,dl=True,dl_dir='./',use_visual_output=False,resize=(256,256))


def simple_cat_dl(thecat,collection,db,dl=True,dl_dir='./',use_visual_output=False,resize=(256,256),min_items=1000,dont_overwrite=True):
    cursor = db[collection].find({'categories':thecat})
    count = cursor.count()
    print('category {} has {} items'.format(thecat,count))
    if count<min_items:
        print('too few items ({} < {}'.format(count,min_items))
    #no need for this other than here and will cause problems elssewhere
    from tqdm import tqdm
    image_size_levels = ['Small','Medium','Large','XLarge','Original','Best']
    size_level=len(image_size_levels)-1
    attempts=0
    succesful_dl=0
    unsuccesful_dl=0
    no_doc=0
    no_images=0
    no_known_size=0
    n_hashed_names=0
    n_already_dl=0
    for doc in tqdm(cursor):
#            while doc is not None:
#                print('doc:'+str(doc))
        attempts+=1
        if doc is None:
            print('got none doc')
            no_doc+=1
            continue
        if not 'images' in doc:
            print('no images field in doc')
            no_images+=1
            continue
#                print('images:'+str(doc['images']))
        img_url=None
        while img_url==None and size_level>=0:
            if image_size_levels[size_level] in doc['images']:
                print('{} in images'.format(image_size_levels[size_level]),end='')
                img_url = doc['images'][image_size_levels[size_level]]
                break
            else:
                print('NO {} in images'.format(image_size_levels[size_level]))
            size_level-=1
        if img_url == None:
            print('couldnt get image of any size (level ='+str(size_level)+' url:'+str(img_url))
            print('images:'+str(doc['images']))
            no_known_size += 1
            raw_input('ret to cont')
            continue
        if dl:
            if not 'id' in doc:
                hash = hashlib.sha1()
                hash.update(str(time.time()))
                id =  str(hash.hexdigest()[:10])
                print('doc has no id, using random {}'.format(id))
                n_hashed_names +=1
            else:
                id = str(doc['id'])
            save_dir = os.path.join(dl_dir,collection.lower())
            Utils.ensure_dir(save_dir)
            save_dir = os.path.join(save_dir,thecat.lower())
            Utils.ensure_dir(save_dir)
            save_name = os.path.join(save_dir,id+'.jpg')
            print('saving to '+str(save_name),end='')
            if dont_overwrite:
                if not os.path.exists(save_name): #get image unless its alreadyhere
                    img_arr = Utils.get_cv2_img_array(img_url)
                    cv2.imwrite(save_name,img_arr)
                else:
                    print('image {} exists'.format(save_name))
                    n_already_dl +=1
                    continue
            if img_arr is None:
                print('WARNING!! did not get image from '+str(img_url))
                unsuccesful_dl = 1
                continue
            succesful_dl+=1
            incoming_size = img_arr.shape[0:2]
            if incoming_size[0]>2*resize[0] and incoming_size[1]>2*resize[1]: #size way bigger than needed
                size_level=max(size_level-1,0)
                print('adjusting size level down to '+str(size_level))
            if incoming_size[0]<resize[0] or incoming_size[1]<resize[1]: #size smaller than needed
                size_level=min(size_level+1,len(image_size_levels)-1)
                print('adjusting size level up to '+str(size_level))
            if resize is not None:
                img_arr = imutils.resize_keep_aspect(img_arr,output_size=resize)

            if use_visual_output:
                cv2.imshow('img',img_arr)
                cv2.waitKey(10)
            cv2.imwrite(save_name,img_arr)
        else:
            print('doc not being saved')
        print('attempts {}/{} success {} un {} no_images {} nodoc {} nosize {} nhash {} n_already {}'.format(attempts,count,succesful_dl,unsuccesful_dl,no_images,no_doc,no_known_size,n_hashed_names,n_already_dl))

    print('finished')
    dict={'cat':thecat,
        'collection':collection,
        'attempts':attempts,
        'succesful_dl':succesful_dl,
        'unsuccesful_dl':unsuccesful_dl,
        'no_doc':no_doc,
        'no_images':no_images,
        'no_known_size':no_known_size,
        'n_hashed_names':n_hashed_names,
        'n_already_dl':n_already_dl}


    with open('dbdata.txt','a') as fp:
        json.dump(dict,fp,indent=4)


def simple_kw_find(category,db='ShopStyle_Female',check_all_dbs=True):
    for thedb in ['ShopStye_Female','ShopStyle_Male','amazon_US_Male','amazon_DE_Female',
                  'amaze_Male','Amazon_US_Female_archive']:
        c = db.thedb.find({"categories":category})

#db.ShopStyle_Female.distinct("categories")

def enqueue_for_download(q, iterable, feature_name, category_id, max_images=100000):
    job_results = []
    for prod in iterable:
        res = q.enqueue(download_image, prod, feature_name, category_id, max_images)
        job_results.append(res.result)
    return job_results

def download_image(prod, feature_name, category_id, max_images):
    downloaded_images = 0
    directory = os.path.join(category_id, feature_name)
    try:
        downloaded_images = len([name for name in os.listdir(directory) if os.path.isfile(name)])
    except:
        pass
    if downloaded_images < max_images:
            xlarge_url = prod['image']['sizes']['XLarge']['url']

            img_arr = Utils.get_cv2_img_array(xlarge_url)
            if img_arr is None:
                logging.warning("Could not download image at url: {0}".format(xlarge_url))
                return

            relevance = background_removal.image_is_relevant(img_arr)
            if relevance.is_relevant:
                logging.info("Image is relevant...")

                filename = "{0}_{1}.jpg".format(feature_name, prod["id"])
                filepath = os.path.join(directory, filename)
                Utils.ensure_dir(directory)
                logging.info("Attempting to save to {0}...".format(filepath))
                success = cv2.imwrite(filepath, img_arr)
                if not success:
                    logging.info("!!!!!COULD NOT SAVE IMAGE!!!!!")
                    return 0
                # downloaded_images += 1
                logging.info("Saved... Downloaded approx. {0} images in this category/feature combination"
                             .format(downloaded_images))
                return 1
            else:
                # TODO: Count number of irrelevant images (for statistics)
                return 0

def run(category_id, search_string_dict=None, async=True):
    logging.info('Starting...')
    download_images_q = Queue('download_images', connection=redis_conn, async=async)
    search_string_dict = search_string_dict or descriptions_dict

    job_results_dict = dict.fromkeys(descriptions_dict)

    for name, search_string_list in search_string_dict.iteritems():
        for search_string in search_string_list:
            cursor = find_products_by_description(search_string, category_id, name)
            job_results_dict[name] = enqueue_for_download(download_images_q, cursor, name, category_id, MAX_IMAGES)

    while True:
        time.sleep(10)
        for name, jrs in job_results_dict.iteritems():
            logging.info(
                "{0}: Downloaded {1} images...".format(name,
                                                       sum((job.result for job in jrs if job and job.result))))

def download_all_images_in_category(category_id,download_dir):
    cursor = find_products_by_category(category_id)
    Utils.ensure_dir(download_dir)
    count = 0
    for prod in cursor:
        xlarge_url = prod['image']['sizes']['XLarge']['url']
        img_arr = Utils.get_cv2_img_array(xlarge_url)
        if img_arr is None:
            logging.warning("Could not download image at url: {0}".format(xlarge_url))
            continue
        filename = "{0}_{1}.jpg".format(category_id, prod["id"])
        filepath = os.path.join(download_dir, filename)
        logging.info("Attempting to save to {0}...".format(filepath))
        success = cv2.imwrite(filepath, img_arr)
        if not success:
            logging.warning("!!!!!COULD NOT SAVE IMAGE!!!!!")
            continue
        count += 1
        logging.info("Saved... Downloaded approx. {0} images in this category/feature combination"
                     .format(count))

def get_shopstyle_nadav(download_dir='./',max_images_per_cat = 1000):
    '''
    dl shopstyle images, grabcut out the database image, and make a mask with category number
    currently, ladies only
    :return:
    '''
    cats = constants.paperdoll_relevant_categories
    for cat in cats:
        #women only right now
        print('category:{} number {}'.format(cat,cat_count))
        cursor = db.ShopStyle_Female.find({'categories': cat})
        count =0
        cat_count = 0
        for prod in cursor:
            xlarge_url = prod['images']['XLarge']
            img_arr = Utils.get_cv2_img_array(xlarge_url)
            if img_arr is None:
                logging.warning("Could not download image at url: {0}".format(xlarge_url))
                continue
            filename = "{0}_{1}.jpg".format(cat, prod["id"])
            filepath = os.path.join(download_dir, filename)
            logging.info("Attempting to save to {0}...".format(filepath))
            success = cv2.imwrite(filepath, img_arr)
            if not success:
                logging.warning("!!!!!COULD NOT SAVE IMAGE!!!!!")
                continue
            logging.info("Saved... Downloaded approx. {0} images in this category/feature combination"
                         .format(count))
            h,w = img_arr.shape[:2]
            rect = [20, 20, w-40, h-40]
            bgfgmask = cat_count * bgfgmask / 255
            bgfgmask = bgfgmask.astype(np.uint8)

            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
            return mask2

            grabmask = background_removal.simple_mask_grabcut(img_arr,rect=None ,mask=bgfgmask)


            maskname = "{0}_{1}_mask.png".format(cat, prod["id"])
            success = cv2.imwrite(maskname, grabmask)
            if not success:
                logging.warning("!!!!!COULD NOT SAVE IMAGE!!!!!")
                continue
            cat_count = cat_count + 1
            count = count + 1
            if count>max_images_per_cat:
                break

def display_shopstyle_nadav(download_dir='./'):
    cats = constants.paperdoll_relevant_categories
    n_cats = len(cats)
    count = 0
    images_only = [f for f in os.listdir(download_dir) if 'jpg' in f and not '_mask' in f]
    print('{} jpg images without _mask in the name'.format(len(images_only)))
    for imagefile in images_only:
        full_imagename = os.path.join(download_dir,imagefile)
        img_arr = Utils.get_cv2_img_array(full_imagename)
        if img_arr is None:
            logging.warning("Could not open image at : {0}".format(full_imagename))
            continue
        corresponding_mask = full_imagename[0:-4] + '_mask.png'
        mask_arr = Utils.get_cv2_img_array(corresponding_mask)
        mask_arr = mask_arr.astype(np.uint8)
        if mask_arr is None:
            logging.warning("Could not mask at : {0}".format(corresponding_mask))
            continue
        factor = 255/n_cats
        mask_img = np.multiply(mask_arr,255)
        mask_img = mask_img.astype(np.uint8)
        h,w = img_arr.shape[:2]
        combined_img = np.zeros([h,w*2,3])
        combined_img[:,0:w,:] = img_arr
        combined_img[:,w:,:] = mask_img
        cv2.imshow(imagefile+'comb',combined_img)
        cv2.imshow(imagefile+'orig',img_arr)
        cv2.imshow(imagefile+'mask',mask_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print('file: {} uniques:{} count:{}'.format(imagefile,np.unique(mask_arr),count))
        count = count + 1


def fix_shopstyle_nadav(download_dir='./'):
    '''rab
    dl shopstyle images
    currently, ladies only
    :return:
    '''
    images_only = [f for f in os.listdir(download_dir) if 'jpg' in f and not '_mask' in f]
    print('{} jpg images without _mask in the name'.format(len(images_only)))
    cats = constants.paperdoll_relevant_categories
    count = 0
    for imagefile in images_only:
        catno = - 1
        for cat in cats:
            if cat in imagefile:
                catno = cats.index(cat)
                print('img: {} category:{}'.format(imagefile,catno))
                break
        if catno == -1:
            logging.warning('could not find cat of image:'+str(imagefile))
            continue

        fullname = os.path.join(download_dir,imagefile)
        img_arr = Utils.get_cv2_img_array(fullname)
        if img_arr is None:
            logging.warning("Could not open image at : {0}".format(imagefile))
            continue
        h,w = img_arr.shape[:2]
        bgmargin_w = int(w/10.0)
        bgmargin_h = int(h/10.0)
        fgmargin_w = int(w/5.0)
        fgmargin_h = int(h/5.0)
        rect = (bgmargin_w, bgmargin_h, w-bgmargin_w*2, h-bgmargin_h*2) #anthing outside rect is obvious backgnd
        #mask = np.zeros(img.shape[:2],np.uint8)
        input_mask = np.zeros((h,w),np.uint8)
#        input_mask = input_mask * cv2.GC_PR_BGD
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)

#        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)


        cv2.grabCut(img_arr, input_mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
        grabmask1 = np.copy(input_mask)

        input_mask[bgmargin_h:-bgmargin_h,bgmargin_w:-bgmargin_w] = cv2.GC_PR_BGD
        input_mask[fgmargin_h:-fgmargin_h,fgmargin_w:-fgmargin_w] = cv2.GC_PR_FGD
        cv2.grabCut(img_arr, input_mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_MASK)
        grabmask2 = np.copy(input_mask)

        print('uniques:'+str(np.unique(grabmask1)))
        print('uniques2:'+str(np.unique(grabmask2)))
        maskname = imagefile.split('.jpg')[0]+'_mask.png'
        success = cv2.imwrite(maskname, grabmask1)
        if not success:
            logging.warning("!!!!!COULD NOT SAVE IMAGE!!!!!")
            continue
        count = count + 1

        outmask1 = np.where((grabmask1==cv2.GC_BGD)|(grabmask1==cv2.GC_PR_BGD),0,1).astype('uint8')
        outimg1 = img_arr*outmask1[:,:,np.newaxis]
        outmask2 = np.where((grabmask2==cv2.GC_BGD)|(grabmask2==cv2.GC_PR_BGD),0,1).astype('uint8')
        outimg2 = img_arr*outmask2[:,:,np.newaxis]

        mask_multiplied = grabmask1*50
        mask_multiplied2 = grabmask2*50
        cv2.imshow('mask',outimg1)
        cv2.imshow('mask2',outimg2)
        cv2.imshow('orig',img_arr)
        cv2.waitKey(0)


def print_logging_info(msg):
    print(msg)


# hackety hack
logging.info = print_logging_info
current_directory_name = os.getcwd()
my_path = os.path.dirname(os.path.abspath(__file__))
MAX_IMAGES = 10000

if __name__ == '__main__':
    fix_shopstyle_nadav(download_dir='./')

    womens = ['womens-accessories',
     'womens-athletic-clothes',
     'bridal',
     'jeans',
     'dresses',
     'womens-intimates',
     'jackets',
     'jewelry',
     'maternity-clothes',
     'womens-outerwear',
     'womens-pants',
     'petites',
     'plus-sizes',
     'shorts',
     'skirts',
     'womens-suits',
     'sweaters',
     'swimsuits',
     'sweatshirts',
     'teen-girls-clothes',
     'womens-tops']

    mens = ['mens-accessories',
     'mens-athletic',
     'mens-big-and-tall',
     'mens-jeans',
     'mens-outerwear',
     'mens-pants',
     'mens-shirts',
     'mens-shorts',
     'mens-sleepwear',
     'mens-blazers-and-sport-coats',
     'mens-suits',
     'mens-sweaters',
     'mens-sweatshirts',
     'mens-swimsuits',
     'mens-ties',
     'teen-guys-clothes',
     'mens-underwear-and-socks',
     'mens-watches-and-jewelry']


    all_shopstyle = womens+mens
    dl_dir = '/home/jeremy/shopstyle_images'
    for cat in all_shopstyle:
        full_dl_dir = os.path.join(dl_dir,cat)
        Utils.ensure_dir(full_dl_dir)
        download_all_images_in_category(cat,full_dl_dir)


#db.collection_names
#ShopStyle Men, Women
#shopstyle amazon ebay from most to least accurate
#c = db.ShopStyle_Female.find({"categories":"dress"})

#db.ShopStyle_Female.distinct("categories")
'''Out[58]:
[u'blazer',
 u'blouse',
 u'cardigan',
 u'coat',
 u'dress',
 u'jacket',
 u'jeans',
 u'leggings',
 u'pants',
 u'shirt',
 u'shorts',
 u'skirt',
 u'suit',
 u'sweater',
 u'sweatshirt',
 u't-shirt',
 u'tights',
 u'top',
 u'vest',
 u'vests']

'''''