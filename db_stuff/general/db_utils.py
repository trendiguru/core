# -*- coding: utf-8 -*-

import logging
import hashlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ...constants import db, redis_conn
from datetime import datetime
from ...Yonti import pymongo_utils
from rq import Queue
import sys
from PIL import Image
import numpy as np
from scipy import fftpack
from ...Utils import get_cv2_img_array
from time import sleep
import unicodedata


refresh_q = Queue('refresh', connection=redis_conn)
phash_q = Queue('phash', connection=redis_conn)

today_date = str(datetime.date(datetime.now()))
last_percent_reported = None
count = 1


def progress_bar(blocksize, total, last_cnt=None, last_pct=None):
    global last_percent_reported, count
    last_count = last_cnt or count
    percent = int(last_count*blocksize * (100 / total))

    last_percent = last_pct or last_percent_reported
    if last_percent != percent:
        if percent % 5 == 0:
            sys.stdout.write(' %s%% ' % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write('#')
            sys.stdout.flush()

        last_percent_reported = percent
        count += 1

    return percent


def print_error(title, message=''):
    title_len = len(title)
    if title_len > 70:
        m = ''
    else:
        minus_len = int(80 - title_len / 2)
        m = '-'
        for i in range(minus_len):
            m += '-'

    dotted_line = '%s %s %s' % (m, title, m)

    if type(message) != str:
        message = str(message)
    message_len = len(message)
    if message_len > 0:
        print ('%s' % dotted_line)
        print (message)
    print ('%s' % dotted_line)


def log2file(mode, log_filename, message='', print_flag=True):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_filename, mode=mode)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    if type(message) == unicode:
        message = unicodedata.normalize('NFKD', message).encode('ascii', 'ignore')
    if type(message) != str:
        message = str(message)
    if len(message):
        logger.info(message)
        logger.removeHandler(handler)
        del logger, handler
        if print_flag:
            print_error(message)
    else:
        return logger, handler


def binary_array_to_hex(arr):
    h = 0
    s = []
    for i, v in enumerate(arr.flatten()):
        if v:
            h += 2**(i % 8)
        if (i % 8) == 7:
            s.append(hex(h)[2:].rjust(2, '0'))
            h = 0
    return "".join(s)


def get_p_hash(image, hash_size=16, img_size=16):
    image = Image.fromarray(image)
    image = image.convert("L").resize((img_size, img_size), Image.ANTIALIAS)
    pixels = np.array(image.getdata(), dtype=np.float).reshape((img_size, img_size))
    dct = fftpack.dct(fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    flat = diff.flatten()
    hexa = binary_array_to_hex(flat)
    return hexa


def get_p_hash_testing(image, hash_size=16, img_size=(16,16)):
    image = Image.fromarray(image)
    image = image.convert("L").resize((img_size), Image.ANTIALIAS)
    pixels = np.array(image.getdata(), dtype=np.float).reshape(img_size)
    dct = fftpack.dct(fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    flat = diff.flatten()
    hexa = binary_array_to_hex(flat)
    return hexa, int(hexa,16)


def get_hash(image):
    m = hashlib.md5()
    m.update(image)
    url_hash = m.hexdigest()
    return url_hash


def thearchivedoorman(col_name, instock_limit=2, archive_limit=7):
    """
    clean the archive from items older than a week
    send items to archive
    """
    collection = db[col_name]
    archive_name = col_name+'_archive'
    pymongo_utils.delete_or_and_index(collection_name=archive_name, index_list=['id'])
    collection_archive = db[archive_name]
    archivers = collection_archive.find(no_cursor_timeout=True)
    not_updated = collection.find({"download_data.dl_version": {"$ne": today_date}}, no_cursor_timeout=True)
    out_stockers = collection.find({'status.instock': False}, no_cursor_timeout=True)
    archivers_count = archivers.count()
    not_updated_count = not_updated.count()
    out_stockers_count = out_stockers.count()
    total = archivers_count + not_updated_count + out_stockers_count
    block_size = total/200

    y_new, m_new, d_new = map(int, today_date.split("-"))
    a = n = 0
    for a, item in enumerate(archivers):
        # if a % block_size == 0:
        #     progress_bar(block_size, total)
        y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
        days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
        if days_out < archive_limit:
            collection_archive.update_one({'id': item['id']}, {"$set": {"status.days_out": days_out}})
        else:
            collection_archive.delete_one({'id': item['id']})

    # add to the archive items which were not downloaded in the last 2 days
    progress = a
    for n, item in enumerate(not_updated):
        # if (n+progress) % block_size == 0:
        #     progress_bar(block_size, total)
        y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
        days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
        if days_out > instock_limit:
            collection.delete_one({'id': item['id']})
            existing = collection_archive.find_one({"id": item["id"]})
            if existing:
                continue

            if days_out < archive_limit:
                item['status']['instock'] = False
                item['status']['days_out'] = days_out
                collection_archive.insert_one(item)
    progress = n + a
    # move to the archive all the items which were downloaded today but are out of stock

    for o, item in enumerate(out_stockers):
        # if (o + progress) % block_size == 0:
        #     progress_bar(block_size, total)
        collection.delete_one({'id': item['id']})
        existing = collection_archive.find_one({"id": item["id"]})
        if existing:
            continue
        collection_archive.insert_one(item)

    archivers.close()
    not_updated.close()
    out_stockers.close()
    collection_archive.reindex()
    print('')


def refresh_worker(doc, name, cats=[]):
    from ... import find_similar_mongo
    collection = db.images
    for person in doc['people']:
        gender = person['gender']
        if gender is None:
            gender = 'Female'
        col_name = name + '_' + gender
        for item in person['items']:
            similar_res = item['similar_results']
            if name in similar_res:
                fp = item['fp']
                category = item['category']
                if len(cats) > 0:
                    if category in cats:
                        _, new_similar_results = find_similar_mongo.find_top_n_results(fingerprint=fp,
                                                                                       collection=col_name,
                                                                                       category_id=category,
                                                                                       number_of_results=100)
                    else:
                        new_similar_results = similar_res[name]

                else:
                    _, new_similar_results = find_similar_mongo.find_top_n_results(fingerprint=fp,
                                                                                   collection=col_name,
                                                                                   category_id=category,
                                                                                   number_of_results=100)

                similar_res[name] = new_similar_results
    collection.replace_one({'_id': doc['_id']}, doc)


def refresh_similar_results(name, cats=[]):
    collection = db.images
    query = 'people.items.similar_results.%s' % name
    relevant_imgs = collection.find({query: {'$exists': 1}})
    total = relevant_imgs.count()
    for current, img in enumerate(relevant_imgs):
        refresh_q.enqueue(refresh_worker, doc=img, name=name, cats=cats, timeout=1800)
        print ('%d/%d sent' % (current, total))
    progress = 0
    while refresh_q.count > 0:
        progress_new = (1 - refresh_q.count / float(total))*100
        if progress != progress_new:
            msg = '%.2f%% done' % progress_new
            progress = progress_new
            print (msg)

    print ('REFRESH DONE!')


def get_indexes_names(coll):
    idx_info = coll.index_information()
    keys = idx_info.keys()
    keys.remove('_id_')
    # removes the '_1' from the key names
    keys = [k[:-2] for k in keys]
    print (keys)
    return keys


def reindex(collection_name, new_indexes=None):
    collection = db[collection_name]
    current_keys = get_indexes_names(collection)
    oldindexes = new_indexes or current_keys
    for index in oldindexes:
        print (index)
        if new_indexes is None:
            collection.drop_index(index)
        collection.create_index(index, background=True)
    print('Index done!')


def phash_worker(col_name, url, idx):
    collection = db[col_name]
    image = get_cv2_img_array(url)
    if image is None:
        collection.delete_one({'_id': idx})
        return
    p_hash = get_p_hash(image)
    p_hash_exists = collection.find_one({'p_hash': p_hash})
    if p_hash_exists:
        print('p_hash exists')
        return
    collection.update_one({'_id': idx}, {'$set': {'p_hash': p_hash}})
    print ('p_hash added')
    return


def p_hash_many(col_name, redo_all=False):
    collection = db[col_name]
    col_indexes = get_indexes_names(collection)
    if 'p_hash' not in col_indexes:
        collection.create_index('p_hash', background=True)

    x = 0
    all_count = 1
    while x < all_count:
        try:
            if redo_all:
                redo_all = False
                all_items = collection.find({}, {'images.XLarge': 1})
            else:
                all_items = collection.find({'p_hash': {'$exists': 0}}, {'images.XLarge': 1})
            x = 0
            all_count = all_items.count()
            for x, item in enumerate(all_items):
                if x % 100 == 0:
                    print ('%d/%d' % (x, all_count))

                url = item['images']['XLarge']
                idx = item['_id']
                phash_q.enqueue(phash_worker, args=(col_name, url, idx), timeout=1800)
                while phash_q.count > 50000:
                    sleep(300)

            while phash_q.count > 0:
                sleep(60)
            break

        except Exception as e:
            print (e.message)
            while phash_q.count > 0:
                sleep(60)


    print_error('clear duplicates')
    all_updated = collection.find({}, {'p_hash': 1})
    all_count = all_updated.count()
    for x, item in enumerate(all_updated):
        if x % 100 == 0:
            print ('%d/%d' % (x, all_count))
        idx = item['_id']
        keys = item.keys()
        if 'p_hash' not in keys:
            collection.delete_one({'_id': idx})
            continue
        p_hash = item['p_hash']
        idx = item['_id']
        p_hash_exists = collection.find_one({'p_hash': p_hash, '_id': {'$ne': idx}})
        if p_hash_exists:
            print('p_hash exists')
            collection.delete_many({'p_hash': p_hash, '_id': {'$ne': idx}})
            continue
    msg = '%s p_hash done!' % col_name
    print_error(msg)


def email(col_name, title='DONE'):
    yonti = 'yontilevin@gmail.com'
    sender = 'Notifier@trendiguru.com'
    # recipient = 'members@trendiguru.com'

    # Open a plain text file for reading.  For this example, assume that
    msg = MIMEMultipart('alternative')
    msg['Subject'] = '%s download %s!' %(col_name, title)
    msg['From'] = sender
    msg['To'] = yonti

    txt2 = '<h3> do something with it </h3>\n'

    html = """\
    <html>
    <head>
    </head>
    <body>"""
    html = html + txt2 + """
    </body>
    </html>
    """
    part1 = MIMEText(html, 'html')
    msg.attach(part1)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # server.set_debuglevel(True)  # show communication with the server
    try:
        server.login('yonti0@gmail.com', "Hub,hKuhiPryh")
        server.sendmail(sender, [yonti], msg.as_string())
        print "sent"
    except:
        print "error"
    finally:
        server.quit()


categories_badwords = ['SLEEPWEAR', 'SHAPEWEAR', 'SLIPS', 'BEDDING', 'LINGERIE', 'CAMISOLES', 'JEWELRY', 'SPORTS',
                       'WATCHES', 'PERFUMES', 'COLOGNES', 'HEALTH', 'TOYS', 'SUNGLASSES', 'COSMETICS', 'LUGGAGE',
                       'MEDIA', 'SUPPLIES', 'MUSICAL', 'PENS', 'PENCILS', 'COMPACTS', 'VASES', 'PANTIES', 'PAJAMAS',
                       'ROBES', 'SOCKS', 'HATS', 'HEADWEAR', 'ARTS', 'CELLULAR', 'CHILDREN', 'SCARVES', 'GLOVES',
                       'WINTER_GLOVES', 'HANDBAGS', 'KITCHEN', 'ELECTRONICS', 'BIKE', 'UNDERWEAR', 'FURNITURE',
                       'BIRDS', 'KEYCHAINS', 'CHEER', 'POM', 'NIGHTGOWNS', 'CHEMISES', 'SCRUB', 'SCRUBS',
                       'BATHROBES',
                       'BRACELETS', 'PANTYHOSE', 'CHILDRENS', 'WRISTLETS', 'CLUTCHES', 'INTIMATES', 'SCHOOL',
                       'PERFUME', 'EARRINGS', 'ANKLETS', 'WALLETS', 'RINGS', 'KIMONO', 'DOGS', 'CATS', 'VEHICLES',
                       'SCARF', 'PONCHO', 'RING', 'GOWN', 'HIDDEN', 'SHAWLS', 'RAINCOATS', 'GRAPHICS', 'GYMGO',
                       'AUTO',
                       'WIGS', 'DEFAULT', 'MANNEQUINS', 'FORMS', 'HARDWARE', 'SLEEP', 'MATERNITY', 'EARMUFFS',
                       'EARWARMERS', 'VEILS', 'HANDKERCHIEFS', 'PHONES', 'MITTENS', 'UMBRELLAS', 'CYCLING',
                       'HEADPIECES', 'DOCTORS', 'PET', 'ANIMALS', 'SKATING', 'BRAS', 'BRA', 'HOSIERY', 'INFANT',
                       'BABY',
                       'GIRLS', 'GYMBOREE', 'BRIEFS', 'BRIEF', 'WATCH', 'GIRL', 'PANTY', 'THONG', 'BOYSHORTS',
                       'KNICKER', 'HAT', 'ELECTRIC', 'FLAME', 'KNICKERS', 'SCOOPS', 'LACE', 'BOYSHORT', 'FLEECE',
                       'SHORTIES', 'SHORTIE', 'NIGHTGOWN', 'UNDERWIRE', 'SKIMMERS', 'SKIMMER', 'LABCOAT',
                       'HIPSTER', 'HIPSTERS']


categories_keywords = ['BELT', 'BELTS', 'BIKINI', 'BIKINIS', 'BLAZER', 'BLAZERS', 'BLOUSE', 'BLOUSES', 'BOOTS', 'CAPRI',
                       'CAPRIS', 'CARDIGAN', 'COAT', 'COATS', 'DRESS', 'DRESSES', 'HILLS', 'HOODIE', 'HOODIES',
                       'JACKET',
                       'JACKETS', 'JEANS', 'JUMPSUIT', 'JUMPSUITS', 'KNIT', 'KNITS', 'LEGGING', 'LEGGINGS', 'LOAFERS',
                       'MOCCASINS', 'OXFORDS', 'PANT', 'PANTS', 'POLO', 'ROMPER', 'ROMPERS', 'SANDALS', 'BLOUSON',
                       'SHIRT', 'SHIRTS', 'SHOES', 'SHORTS', 'SKIRT', 'SKIRTS', 'SNIKERS', 'STOCKINGS', 'SUIT', 'SUITS',
                       'SWEATERS', 'SWEATSHIRT', 'SWEATSHIRTS', 'SWIM', 'SWIMING', 'SWIMINGSUIT', 'OXFORD', 'KICKS',
                       'SWEATER', 'SWIMINGSUITS', 'SWIMSUIT', 'SWIMSUITS', 'T', 'TEE', 'TEES', 'TIGHTS', 'TOP', 'TOPS',
                       'TROUSER', 'TROUSERS', 'TSHIRT', 'TSHIRTS', 'TUNIC', 'TUNICS', 'VEST', 'VESTS', 'MAXIDRESS',
                       'MAXI', 'SHIRTDRESS', 'SHEATH', 'SWEATERDRESS', 'SUNDRESS', 'LOUNG', 'LOUNGER', 'SHRUG',
                       'SLACKS',
                       'SLITDRESS', 'SWIMDRESS', 'JUMPER', 'TANK', 'POULOVER', 'PIECEDRESS', 'HOODY', 'TIE', 'CHINOS',
                       'PULLOVER', 'PULLOVERS', 'CHINO', 'BOARDSHORT', 'BOARDSHORTS', 'PANTSUIT', 'CAPTOE', 'TROTTERS',
                       'FLATS', 'DRES', 'MINIDRESS', 'SWEATPANTS', 'JOGGER', 'JEGGINGS', 'JERSEY', 'KHAKIS', 'LOAFER',
                       'SHOE', 'BLUCHER', 'JERSEYS', 'DUNGAREE', 'OVERALL', 'OVERALLS', 'JOGGERS', 'SPORTCOAT', 'PARKA',
                       'SOFTSHELL', 'OVERSHIRT', 'WINDBREAKER', 'WINDSHIRT', 'JEAN', 'SHORT', 'WALKSHORT', 'WALKSHORTS',
                       'JKT', 'SKORT', 'SKIRTINI', 'SWIMWEAR', 'PEACOAT', 'MONIKINI', 'TANKINI', 'RASHGUARD',
                       'MINISKIRT', 'CAMI', 'POPOVER', 'CAMISOLE', 'CARDI']

categories_swap =  {'BELT': 'belt', 'BELTS': 'belt',
                    'BIKINIS': 'bikini', 'BIKINI': 'bikini',
                    'BLAZERS': 'blazer', 'BLAZER': 'blazer',
                    'BLOUSES': 'blouse', 'BLOUSE': 'blouse', 'TUNICS': 'blouse', 'TUNIC': 'blouse',
                    'CAPRIS': 'capris', 'CAPRI': 'capris',
                    'CARDIGANS': 'cardigan', 'CARDIGAN': 'cardigan', 'CARDI': 'cardigan',
                    'COATS': 'coat', 'COAT': 'coat', 'PARKA': 'coat', 'PEACOAT': 'coat',
                    'DRESSES': 'dress', 'DRESS': 'dress', 'MAXIDRESS': 'dress', 'MAXI': 'dress',
                    'DRES': 'dress',
                    'SHIRTDRESS': 'dress', 'SHEATH': 'dress', 'SWEATERDRESS': 'dress', 'SUNDRESS': 'dress',
                    'LOUNG': 'dress', 'LOUNGER': 'dress', 'SHRUG': 'dress', 'SLITDRESS': 'dress',
                    'PIECEDRESS': 'dress',
                    'MINIDRESS': 'dress',
                    'JACKETS': 'jacket', 'JACKET': 'jacket', 'BLOUSON': 'jacket', 'SPORTCOAT': 'jacket',
                    'SOFTSHELL': 'jacket', 'WINDBREAKER': 'jacket', 'JKT': 'jacket',
                    'JEANS': 'jeans', 'JEGGINGS': 'jeans', 'JEAN': 'jeans',
                    'KNITS': 'knits', 'KNIT': 'knits', 'POULOVER': 'knits',
                    'PANTS': 'pants', 'PANT': 'pants', 'TROUSERS': 'pants', 'TROUSER': 'pants',
                    'CHINOS': 'pants',
                    'SLACKS': 'pants', 'CHINO': 'pants', 'SWEATPANTS': 'pants', 'JOGGER': 'pants',
                    'KHAKIS': 'pants',
                    'JOGGERS': 'pants',
                    'POLO': 'shirt',
                    'ROMPERS': 'rompers', 'ROMPER': 'rompers',
                    'SHIRTS': 'shirt', 'SHIRT': 'shirt', 'OVERSHIRT': 'shirt', 'POPOVER': 'shirt',
                    'TANK': 'tanktop', 'CAMI': 'tanktop', 'CAMISOLE': 'tanktop',
                    'JERSEY': 'jersey', 'JERSEYS': 'jersey',
                    'SHOES': 'shoes', 'BOOTS': 'shoes', 'HILLS': 'shoes', 'LOAFERS': 'shoes',
                    'MOCCASINS': 'shoes',
                    'SNEAKERS': 'shoes', 'OXFORDS': 'shoes', 'SANDALS': 'shoes', 'OXFORD': 'shoes',
                    'SHOE': 'shoes', 'BLUCHER': 'shoes',
                    'KICKS': 'shoes', 'CAPTOE': 'shoes', 'TROTTERS': 'shoes', 'FLATS': 'shoes', 'LOAFER': 'shoes',
                    'SHORTS': 'shorts', 'BOARDSHORT': 'shorts', 'BOARDSHORTS': 'shorts', 'SHORT': 'shorts',
                    'WALKSHORT': 'shorts', 'WALKSHORTS': 'shorts',
                    'SKIRTS': 'skirt', 'SKIRT': 'skirt', 'SKORT': 'skirt', 'MINISKIRT': 'skirt',
                    'STOCKINGS': 'stockings',
                    'LEGGINGS': 'leggings', 'LEGGING': 'leggings',
                    'TIGHTS': 'tights',
                    'SUITS': 'suit', 'SUIT': 'suit', 'JUMPSUIT': 'suit', 'JUMPSUITS': 'suit', 'PANTSUIT': 'suit',
                    'SWEATERS': 'sweater', 'SWEATER': 'sweater', 'JUMPER': 'sweater', 'PULLOVER': 'sweater',
                    'PULLOVERS': 'sweater', 'WINDSHIRT': 'sweater', 'FLEECE': 'sweater',
                    'SWEATSHIRTS': 'sweatshirt', 'SWEATSHIRT': 'sweatshirt',
                    'HOODIES': 'sweatshirt', 'HOODIE': 'sweatshirt', 'HOODY': 'sweatshirt',
                    'SWIMSUIT': 'swimsuit', 'SWIMINGSUIT': 'swimsuit', 'SWIMSUITS': 'swimsuit', 'SWIM': 'swimsuit',
                    'SWIMINGSUITS': 'swimsuit', 'SWIMING': 'swimsuit', 'SWIMDRESS': 'swimsuit',
                    'SKIRTINI': 'swimsuit',
                    'SWIMWEAR': 'swimsuit', 'MONIKINI': 'swimsuit', 'TANKINI': 'swimsuit', 'RASHGUARD': 'swimsuit',
                    'T': 't-shirt', 'TSHIRTS': 't-shirt', 'TSHIRT': 't-shirt', 'TEES': 't-shirt', 'TEE': 't-shirt',
                    'TOPS': 'top', 'TOP': 'top',
                    'VESTS': 'vest', 'VEST': 'vest',
                    'TIE': 'tie',
                    'DUNGAREE': 'overall', 'OVERALL': 'overall', 'OVERALLS': 'overall'}

