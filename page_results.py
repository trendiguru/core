
import hashlib
import logging
import datetime
import maxminddb
import tldextract
import bson
from jaweson import msgpack
import requests
from rq import push_connection
# ours
import Utils
import constants
import find_similar_mongo
from .background_removal import image_is_relevant

db = constants.db
start_pipeline = constants.q1
relevancy = constants.q2
manual_gender = constants.q3
lang = ""
image_coll_name = "images"
prod_coll_name = "products"
geo_db_path = '/usr/local/lib/python2.7/dist-packages/maxminddb'
GENDER_ADDRESS = "http://37.58.101.173:8357/neural/gender"
DOORMAN_ADDRESS = "http://37.58.101.173:8357/neural/doorman"
LABEL_ADDRESS = "http://37.58.101.173:8357/neural/label"
geo_reader = maxminddb.open_database(geo_db_path + '/GeoLite2-Country.mmdb')
push_connection(constants.redis_conn)

# -------------------------------------- *** ASYNC-MODE *** ----------------------------------------------

# ----------------------------------------- MAIN-FUNCTIONS -----------------------------------------------


def handle_post(image_url, page_url, products_collection, method):
    # QUICK FILTERS
    # if not db.whitelist.find_one({'domain': domain}):
    #     return False

    if image_url[:4] == "data":
        return False

    if db.iip.find_one({'image_urls': image_url}) or db.irrelevant_images.find_one({'image_urls': image_url}):
        return False

    # IF IMAGE IS IN DB.IMAGES:
    image_obj = db.images.find_one({'image_urls': image_url})
    if image_obj:
        # IF IMAGE HAS RESULTS FROM THIS COLLECTION:
        if has_results_from_collection(image_obj, products_collection):
            methods = [person['segmentation_method'] for person in image_obj['people']]
            if 'nd' in methods:
                image_obj = {'people': [{'person_id': person['_id'], 'face': person['face'],
                             'gender': person['gender']} for person in image_obj['people']],
                             'image_urls': image_url, 'page_url': page_url, 'insert_time': datetime.datetime.now()}
                db.iip.insert_one(image_obj)
                start_pipeline.enqueue_call(func="", args=(page_url, image_url, products_collection, 'pd'),
                                            ttl=2000, result_ttl=2000, timeout=2000)
            return True
        else:
            # ADD RESULTS FROM THIS PRODUCTS-COLLECTION
            add_results_from_collection(image_obj, products_collection)
            return False

    else:
        relevancy.enqueue_call(func=check_if_relevant, args=(image_url, page_url, products_collection, method),
                               ttl=2000, result_ttl=2000, timeout=2000)
        return False


# ---------------------------------------- FILTER-FUNCTIONS ----------------------------------------------

def has_results_from_collection(image_obj, collection):
    for person in image_obj['people']:
        for item in person['items']:
            if collection in item['similar_results']:
                return True
    return False


def is_image_relevant(image_url, collection_name=None):
    collection_name = collection_name or image_coll_name
    if image_url is not None:
        query = {"image_urls": image_url}
        image_dict = db[collection_name].find_one(query, {'relevant': 1, 'people.items.similar_results': 1})
        if not image_dict:
            image = Utils.get_cv2_img_array(image_url)
            if image is None:
                return False
            hash = get_hash(image)
            image_dict = db[collection_name].find_one({'image_hash': hash})
            if image_dict:
                db.images.update_one({'image_hash': hash}, {'$inc': {'views': 1},
                                                            '$addToSet': {'image_urls': image_url}})
                return has_items(image_dict)
        else:
            db.images.update_one(query, {'$inc': {'views': 1}})
            return has_items(image_dict)
    else:
        return False


def has_items(image_dict):
    res = False
    # Easier to ask forgiveness than permission
    # http://stackoverflow.com/questions/1835756/using-try-vs-if-in-python
    try:
        for person in image_dict['people']:
            if 'items' in person.keys():
                for item in person['items']:
                    if 'similar_results' in item.keys():
                        if isinstance(item['similar_results'], list):
                            res = len(item['similar_results']) > 0
                        elif isinstance(item['similar_results'], dict):
                            res = bool(item['similar_results'])
    except:
        pass
    return res


# ----------------------------------------- ROUTE-FUNCTIONS ----------------------------------------------

def get_country_from_ip(ip):
    user_info = geo_reader.get(ip)
    if user_info:
        if 'country' in user_info.keys():
            return user_info['country']['iso_code']
        elif 'registered_country' in user_info.keys():
            return user_info['registered_country']['iso_code']
    else:
        return None


def get_collection_from_ip_and_domain(ip, domain):
    country = get_country_from_ip(ip)
    default_map = constants.which_products_collection['default']
    if domain in constants.which_products_collection.keys():
        domain_map = constants.which_products_collection[domain]
        if country:
            if country in domain_map.keys():
                return domain_map[country]
            elif 'default' in domain_map.keys():
                return domain_map['default']
            else:
                if country in default_map.keys():
                    return default_map[country]
                else:
                    return default_map['default']
        else:
            if 'default' in domain_map.keys():
                return domain_map['default']
            else:
                return default_map['default']
    else:
        if country in default_map.keys():
            return default_map[country]
        else:
            return default_map['default']


def get_collection_from_ip_and_pid(ip, pid='default'):
    country = get_country_from_ip(ip)
    default_map = constants.products_per_ip_pid['default']
    if pid in constants.products_per_ip_pid.keys():
        pid_map = constants.products_per_ip_pid[pid]
        if country:
            if country in pid_map.keys():
                return pid_map[country]
            elif 'default' in pid_map.keys():
                return pid_map['default']
            else:
                if country in default_map.keys():
                    return default_map[country]
                else:
                    return default_map['default']
        else:
            if 'default' in pid_map.keys():
                return pid_map['default']
            else:
                return default_map['default']
    else:
        if country in default_map.keys():
            return default_map[country]
        else:
            return default_map['default']


# ---------------------------------------- PROCESS-FUNCTIONS ---------------------------------------------

def add_results_from_collection(image_obj, collection):
    for person in image_obj['people']:
        for item in person['items']:
            prod = collection + '_' + person['gender']
            fp, similar_results = find_similar_mongo.find_top_n_results(number_of_results=100,
                                                                        category_id=item['category'],
                                                                        fingerprint=item['fp'],
                                                                        collection=prod)
            item['similar_results'][collection] = similar_results
    db.images.replace_one({'_id': image_obj['_id']}, image_obj)


def check_if_relevant(image_url, page_url, products_collection, method):

    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        return

    # Jeremy's neural-doorman

    relevance = image_is_relevant(image, use_caffe=False, image_url=image_url)

    if not relevance.is_relevant:
        hashed = get_hash(image)
        image_obj = {'image_hash': hashed, 'image_urls': [image_url], 'page_urls': [page_url], 'people': [],
                     'relevant': False, 'saved_date': str(datetime.datetime.utcnow()), 'views': 1,
                     'labels': labelize(image)}
        db.irrelevant_images.insert_one(image_obj)
        db.labeled_irrelevant.insert_one(image_obj)
        return image_obj
    image_obj = {'people': [{'person_id': str(bson.ObjectId()), 'face': face.tolist(),
                             'gender': genderize(image, face.tolist())['gender']} for face in relevance.faces],
                 'image_urls': image_url, 'page_url': page_url, 'insert_time': datetime.datetime.now()}
    db.iip.insert_one(image_obj)
    # db.genderator.insert_one(image_obj)
    start_pipeline.enqueue_call(func="", args=(page_url, image_url, products_collection, method),
                                ttl=2000, result_ttl=2000, timeout=2000)


# --------------------------------------------- NNs -----------------------------------------------------


def genderize(image_or_url, face):
    data = msgpack.dumps({"image": image_or_url, "face": face})
    resp = requests.post(GENDER_ADDRESS, data)
    return msgpack.loads(resp.content)
    # returns {'success': bool, 'gender': Female/Male, ['error': the error as string if success is False]}


def labelize(image_or_url):
    try:
        data = msgpack.dumps({"image": image_or_url})
        resp = requests.post(LABEL_ADDRESS, data)
        labels = msgpack.loads(resp.content)["labels"]
        return {key: float(val) for key, val in labels.items()}
    except:
        return []


# ----------------------------------------- GET-FUNCTIONS -----------------------------------------------

def get_data_for_specific_image(image_url=None, image_hash=None, image_projection=None, product_projection=None,
                                max_results=20, lang=None, products_collection='ShopStyle'):
    """
    this just checks db for an image or hash. It doesn't start the pipeline or update the db
    :param image_url: url of image to find
    :param image_hash: hash (of image) to find
    :return:
    """
    if lang:
        set_lang(lang)
    image_collection = db[image_coll_name]
    # domain = tldextract.extract(page_url).registered_domain
    # products_collection = get_collection_from_ip_and_domain(ip, domain)
    print "##### image_coll_name: " + image_coll_name + " #####"
    # REMEMBER, image_obj is sparse, similar_results have very few fields.
    image_projection = image_projection or {
        '_id': 1,
        'image_hash': 1,
        'image_urls': 1,
        'page_urls': 1,
        'people.gender': 1,
        'people.items.category': 1,
        'people.items.category_name': 1,
        'people.items.item_id': 1,
        'people.items.item_idx': 1,
        'people.items.similar_results.{0}'.format(products_collection): {'$slice': max_results},
        'people.items.similar_results.{0}._id'.format(products_collection): 1,
        'people.items.similar_results.{0}.id'.format(products_collection): 1,
        # 'people.items.svg_url': 1,
        'relevant': 1}

    product_projection = product_projection or {
        #'seeMoreUrl': 1,
        'image.sizes.XLarge.url': 1,
        'images.XLarge': 1,
        'images.Medium': 1,
        'clickUrl': 1,
        #'retailer': 1,
        #'currency': 1,
        'brand': 1,
        # 'brand.localizedName': 1,
        #'description': 1,
        'price.price': 1,
        #'categories': 1,
        'shortDescription': 1,
        #'sizes': 1,
        #'pageUrl': 1,
        '_id': 1,
        'id': 1,
        'price.currency': 1,
    }

    if image_url is None and image_hash is None:
        print "page_results.get_data_for_specific_image wasn't given one of image url or image hash"
        return None
    if image_url is not None:
        print 'looking for image ' + image_url + ' in db '
        query = {"image_urls": image_url}
    else:
        print 'looking for hash ' + image_hash + ' in db '
        query = {"image_hash": image_hash}

    sparse_image_dict = image_collection.find_one(query, image_projection)
    if sparse_image_dict is not None:
        logging.debug('found image (or hash) in db ')
        # hash gets checked in update_image_in_db(), alternatively it could be checked here
        full_image_dict = load_similar_results(sparse_image_dict, product_projection, products_collection)
        merged_dict = merge_items(full_image_dict)
        return merged_dict
    else:
        logging.debug('image / hash  was NOT found in db')
        return None


def load_similar_results(sparse, projection_dict, product_collection_name):
    print "Will load similar results from collection: " + product_collection_name
    for person in sparse["people"]:
        if 'gender' in person.keys():
            collection = db[product_collection_name + '_' + person['gender']]
        else:
            collection = db[product_collection_name + "_Female"]
        if 'items' in person.keys():
            for item in person["items"]:
                # similar_results = []
                ids = [result['id'] for result in item["similar_results"][product_collection_name]]
                similar_results = list(collection.find({"id": {"$in": ids}}, projection_dict))
                for full_result in similar_results:
                    full_result['redirection_path'] = '/' + product_collection_name + '_' +\
                                                     person['gender'] + '/' + str(full_result['_id'])
                # for result in item["similar_results"][product_collection_name]:
                #     full_result = collection.find_one({"id": result["id"]}, projection_dict)
                #     if full_result:
                #         full_result['redirection_path'] = '/' + product_collection_name + '_' +\
                #                                      person['gender'] + '/' + str(full_result['_id'])
                #         similar_results.append(full_result)
                item["similar_results"] = similar_results
    return sparse


def merge_items(doc):
    # doc['items'] = [item for person in doc['people'] for item in person["items"] if 'items' in person.keys()]
    doc['items'] = []
    for person in doc['people']:
        if 'items' in person.keys():
            for item in person['items']:
                doc['items'].append(item)
    del doc["people"]
    return doc


# -------------------------------------------- OTHERS ---------------------------------------------------

def set_lang(new_lang):
    global lang
    global image_coll_name
    global prod_coll_name

    if not new_lang:
        image_coll_name = "images"
        prod_coll_name = "products"
        return image_coll_name
    else:
        lang = new_lang
        lang_suffix = "_" + new_lang
        image_coll_name = "images{0}".format(lang_suffix)
        prod_coll_name = "products{0}".format(lang_suffix)
        return image_coll_name


def get_hash(image):
    m = hashlib.md5()
    m.update(image)
    url_hash = m.hexdigest()
    return url_hash


def get_hash_of_image_from_url(image_url):
    if image_url is None:
        logging.warning("Bad image url!")
        return None
    img_arr = Utils.get_cv2_img_array(image_url)
    if img_arr is None:
        logging.warning('couldnt get img_arr from url:' + image_url + ' in get_hash_of_image')
        return None
    return get_hash(img_arr)
