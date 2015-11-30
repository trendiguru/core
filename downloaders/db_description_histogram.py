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

from trendi.constants import db
from trendi.constants import redis_conn
import trendi.Utils as Utils
import trendi.background_removal as background_removal
from trendi.find_similar_mongo import get_all_subcategories

current_directory_name = os.getcwd()
my_path = os.path.dirname(os.path.abspath(__file__))

# download_images_q = Queue('download_images', connection=redis_conn)  # no args implies the default queue
logging.basicConfig(level=logging.WARNING)
MAX_IMAGES = 10000
descriptions = ['round collar', 'bow collar',
                'ribbed round neck', 'rollneck',
                'slash neck']
# LESSONS: CANNOT PUT MULTIPLE PHRASES IN $text
# v-neck is a superset of v-neckline
descriptions_dict = {'bowcollar': ["\"bow collar\"", "bowcollar"],
                     'crewneck': ["\"crew neck\"", "crewneck", "\"classic neckline\""],
                     'roundneck': ["\"round neck\"", "roundneck"],
                     'scoopneck': ["\"scoop neck\"", "scoopneck"],
                     'squareneck': ["\"square neck\"", "squareneck"],
                     'v-neck': ["\"v-neck\"", "\"v neck\"", "vneck"]}


def get_db_fields(collection='products'):
    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    cursor = db.products.find()
    print('returned cursor')
    if cursor is None:  # make sure training collection exists
        print('couldnt get cursor ' + str(collection))
        return {"success": 0, "error": "could not get collection"}
    doc = next(cursor, None)
    i = 0
    while doc is not None:
        print('checking doc #' + str(i + 1))
        for topic in doc:
            try:
                print(str(topic))
            except UnicodeEncodeError:
                print('unicode encode error')
        i = i + 1
        doc = next(cursor, None)
        print('')
        raw_input('enter key for next doc')
    return {"success": 1}

def collect_description(search_string='pants',category_id='dresses',cutoff=5000):
    cursor = find_products_by_category(category_id)
    if cursor is None:  # make sure training collection exists
        print('couldnt get cursor ' + str(collection))
        return {"success": 0, "error": "could not get collection"}
    doc = next(cursor, None)
    i = 0
    count = cursor.count()
    max_items = 10000000
    max_items = min(max_items,cursor.count())
    check_freq = max(1,max_items/50)
    print(check_freq)
    word_frequencies={}
    while i<max_items and doc is not None:
#        print('checking doc #' + str(i + 1))
        if 'categories' in doc:
            try:
                #print('cats:' + str(doc['categories']))
                pass
            except UnicodeEncodeError:
                print('unicode encode error in cats')
                s = doc['categories']
               # print(s.encode('utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        if 'description' in doc:
            try:
#                print('desc:' + str(doc['description']))
                words = doc['description']
            except UnicodeEncodeError:
                print('unicode encode error in description')
                s = doc['description']
 #               print(s.encode('utf-8'))
                words = s.encode('utf-8')
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        words = words.lower()  #make lowercase
        words = words.replace('.','')  #lose periods
        words = words.replace('<li>','') #lose those thingies
        words = words.replace('</li>','') #these too
        words = words.replace(',','') #these too
        words = words.replace('-','') #these too
        words = words.replace('/','') #these too
        words = words.replace(':','') #these too
        words = words.replace(';','') #these too
        individual_words = words.split()
        for word in individual_words:
            if word in word_frequencies:
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1
#       print(word_frequencies)

        i = i + 1
        doc = next(cursor, None)
        if i % check_freq==0:
            print('{0} of {1} done'.format(i,max_items))
#        raw_input('enter key for next doc')
    sorted_freqs=list(reversed(sorted(word_frequencies.items(), key=itemgetter(1))))
    #sorted_freqs = sorted(word_frequencies, key=lambda word: word[0])  #doesn't give both key and value
    sorted_freqs = purge_common(sorted_freqs)
#    print('sorted:')
#    print(sorted_freqs)
    word_frequencies_filename='word_frequencies'+category_id+'.txt'
    with open(word_frequencies_filename, "w") as outfile:
        print('succesful open, attempting to write word freqs to:'+word_frequencies_filename)
        json.dump(sorted_freqs,outfile, indent=4)
    plot_word_hist(sorted_freqs,category=category_id,cutoff=cutoff)
#    integrate_freqs(sorted_freqs,category=category_id)
#supress giant output
#    return sorted_freqs

def purge_common(unsorted):
    purge_list=['and','a','with','the','in','no','to','at','from','<ul>',
                '</ul>','this','is','for','of','by','on','an','that','a','this',
                'it','you','or','may','true','your','our','only']
#no longer need these as orig list is cleaned up
#    purge_list = [[word,word+'.'] for word in purge_list]
#    purge_list = [elem for l in purge_list for elem in l]  #flatten list comp
#    purge_list = [[word,word.title()] for word in purge_list]
#    purge_list = [elem for l in purge_list for elem in l]  #flatten list comp

    purged = [(entry[0],entry[1]) for entry in unsorted if not entry[0] in purge_list]
    return purged

def integrate_freqs(word_frequencies,category='nocat'):
    integral = []
    normalized = []
    val = 0
    for pair in word_frequencies:
#        print('pair {0} {1} {2}'.format(pair,pair[0],pair[1]))
        n=pair[1]
        val = val + n
        integral.append(val)
    #f=plt.figure(figsize=(20,5))
    for i in range(0,len(integral)):
        normalized.append(integral[i]*1.0/integral[-1])
    x=range(1,len(integral)+1)
    print('total # words='+str(val))
    f=plt.figure(figsize=(20,5))
    ax = f.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.plot(x, normalized,'b.-')
#    ax.set_xticks(x)
 #   ax.set_aspect(1.0)
#    ax.set_xticklabels(labels,rotation='vertical')

#    plt.tight_layout()
 #   plt.savefig(category+'.jpg',bbox_inches='tight')


 #   plt.plot(x,integral)
    plt.title('cumulative percent for '+category)
    plt.grid(True)
    plt.savefig(category+'_cumulative.jpg',bbox_inches='tight')
    return(integral)

def plot_word_hist(word_frequencies,category='nocat',cutoff=1):
#    print('freqs:' +str(word_frequencies))
    labels = [entry[0] for entry in word_frequencies]
    y = [entry[1] for entry in word_frequencies if entry[1]>cutoff]
    x = xrange(len(y))
    x_a = np.arange(len(y))
#    print('x {0} y {1} labels {2}'.format(x,y,labels))
#    f = figure(1)

#    fig, ax = plt.subplots()
#    bar_width = 0.35
#    opacity = 0.4
#    error_config = {'ecolor': '0.3'}
#    rects1 = plt.bar(x_a, y, bar_width,
#                     alpha=opacity,
#                     color='b')

#    plt.xlabel(category)
#    plt.ylabel('frequency')
#    plt.title('word frequencies in '+category+' category')
#    plt.xticks(x_a + bar_width, labels)
#    plt.setp(labels,rotation=90)
    #   plt.set_xticklabels(labels,rotation='vertical')
#   plt.legend()

#    plt.tight_layout()

#    f = plt.figure()
    integral = integrate_freqs(word_frequencies,category=category)
    truncated_integral=integral[0:len(y)]
    f=plt.figure(figsize=(20,5))
    ax = f.add_axes([0.0, 0.0, 1.0, 1.0])
    x_int = xrange(1,len(integral)+1)
#    print('ylength {0} intlengh {1}'.format(len(word_frequencies),len(integral)))
#    print('trunc:'+str(truncated_integral))
    plt.semilogy()
    plt.title('freqs and cumulative count for '+category+' ,cutoff='+str(cutoff)+', '+str(integral[-1])+' total words')
    plt.grid(True)
    ax.plot(x,truncated_integral,'b.-')
#    ax.plot(x, normalized,'b.-')
    ax.bar(x,y)
    ax.set_xticks(x)

 #   ax.set_aspect(1.0)
    ax.set_xticklabels(labels,rotation='vertical')

#    plt.tight_layout()
    plt.savefig(category+'.jpg',bbox_inches='tight')
#    f.show()
#

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
    fields = {"categories": 1, "id": 1, "description": 1}
    cursor = db.products.find(query, fields).batch_size(10)
    logging.info("Found {count} products in cat {category} ".format(count=cursor.count(),
                                                                    category=category_id))
    return cursor


def enqueue_for_download(q, iterable, feature_name, category_id, max_images=MAX_IMAGES):
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

def print_logging_info(msg):
    print msg

# hackety hack
logging.info = print_logging_info

if __name__ == '__main__':
    run()
