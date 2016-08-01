import logging
import hashlib
from ..constants import db, redis_conn
from datetime import datetime
from ..Yonti import pymongo_utils
from rq import Queue
from scipy import fftpack
import itertools
import sys
q = Queue('refresh', connection=redis_conn)
today_date = str(datetime.date(datetime.now()))
last_percent_reported = None
count = 1


def progress_bar(blocksize, total, last_cnt = None, last_pct=None):
    global last_percent_reported, count
    last_count = last_cnt or count
    percent = int(last_count*blocksize *100/ total)

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
    if title_len > 50:
        m = ''
    else:
        minus_len = int(60 - title_len / 2)
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


def get_phash(image):
    pixel_depth = 255.0
    image_data = (image - pixel_depth / 2) / pixel_depth
    dct = fftpack.dct(fftpack.dct(image_data.T, norm='ortho').T, norm='ortho')
    small_dct = dct[0:16, 0:16].tolist()
    pixels = list(itertools.chain.from_iterable(itertools.chain.from_iterable(small_dct)))
    avg = (sum(pixels) - pixels[0]) / (len(pixels) - 1)
    bits = "".join(map(lambda pixel: '1' if pixel > avg else '0', pixels))  # '00010100...'
    hexadecimal = int(bits, 2).__format__('016x').upper()
    return hexadecimal


def get_hash(image):
    m = hashlib.md5()
    m.update(image)
    url_hash = m.hexdigest()
    return url_hash


def theArchiveDoorman(col_name, instock_limit=2, archive_limit=7):
    """
    clean the archive from items older than a week
    send items to archive
    """
    collection = db[col_name]
    archive_name = col_name+'_archive'
    pymongo_utils.delete_or_and_index(collection_name=archive_name, index_list=['id'])
    collection_archive = db[archive_name]
    archivers = collection_archive.find()
    notUpdated = collection.find({"download_data.dl_version": {"$ne": today_date}})
    outStockers = collection.find({'status.instock': False})
    archivers_count = archivers.count()
    notUpdated_count = notUpdated.count()
    outStockers_count = outStockers.count()
    total = archivers_count + notUpdated_count + outStockers_count
    block_size = total/200

    y_new, m_new, d_new = map(int, today_date.split("-"))
    a = n = 0
    for a, item in enumerate(archivers):
        if a % block_size == 0:
            progress_bar(block_size, total)
        y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
        days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
        if days_out < archive_limit:
            collection_archive.update_one({'id': item['id']}, {"$set": {"status.days_out": days_out}})
        else:
            collection_archive.delete_one({'id': item['id']})

    # add to the archive items which were not downloaded in the last 2 days
    progress = a
    for n, item in enumerate(notUpdated):
        if (n+progress) % block_size == 0:
            progress_bar(block_size, total)
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
    for o, item in enumerate(outStockers):
        if (o + progress) % block_size == 0:
            progress_bar(block_size, total)
        collection.delete_one({'id': item['id']})
        existing = collection_archive.find_one({"id": item["id"]})
        if existing:
            continue
        collection_archive.insert_one(item)

    collection_archive.reindex()


def refresh_worker(doc, name):
    from .. import find_similar_mongo
    collection = db.images
    for person in doc['people']:
        gender = person['gender']
        col_name = name + '_' + gender
        for item in person['items']:
            similar_res = item['similar_results']
            if name in similar_res:
                fp = item['fp']
                category = item['category']
                _, new_similar_results = find_similar_mongo.find_top_n_results(fingerprint=fp, collection=col_name, category_id=category,
                                                            number_of_results=100)
                similar_res[name] = new_similar_results
    collection.replace_one({'_id': doc['_id']}, doc)


def refresh_similar_results(name):
    collection = db.images
    query = 'people.items.similar_results.%s' % name
    relevant_imgs = collection.find({query: {'$exists': 1}})
    total = relevant_imgs.count()
    for current, img in enumerate(relevant_imgs):
        q.enqueue(refresh_worker, doc=img, name=name, timeout=1800)
        print ('%d/%d sent' % (current, total))

    while q.count > 0:
        msg = '%.2f done' % (1-q.count/float(total))
        print (msg)

    print ('REFRESH DONE!')


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

categories_swap = { 'BELT': 'belt', 'BELTS': 'belt',
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