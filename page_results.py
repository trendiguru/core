__author__ = 'jeremy'
# MD5 in javascript http://www.myersdaily.org/joseph/javascript/md5-speed-test-1.html?script=jkm-md5.js#calculations
# after reading a bit i decided not to use named tuple for the image structure
# theirs
import hashlib
import logging
import maxminddb
import tldextract
# ours
import Utils
import constants

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
db = constants.db
start_pipeline = constants.q1
lang = ""
image_coll_name = "images"
prod_coll_name = "products"
geo_db_path = '/usr/local/lib/python2.7/dist-packages/maxminddb'


def has_results_from_collection(image_obj, collection):
    for results in image_obj['people'][0]['items'][0]['similar_results']:
        if collection in results.keys():
            return True
    return False


def add_results_from_collection(image_obj, collection):

    return True


def get_collection_from_ip_and_domain(ip, domain):
    if domain in constants.products_per_site.keys():
        return constants.products_per_site[domain]
    else:
        return constants.products_per_site['default']
    # reader = maxminddb.open_database(geo_db_path + '/GeoLite2-Country.mmdb')
    # user_info = reader.get(ip)
    # if user_info:
    #     if 'country' in user_info.keys():
    #         country_iso_code = user_info['country']['iso_code']
    #         return next((k for k, v in constants.products_per_country.items() if country_iso_code in v), None)
    #     elif 'registered_country' in user_info.keys():
    #         country_iso_code = user_info['registered_country']['iso_code']
    #         return next((k for k, v in constants.products_per_country.items() if country_iso_code in v), None)
    # return constants.products_per_country['default']


def route(ip, images_list, page_url):
    ret = {}
    for image_url in images_list:
        # IF IMAGE IS IN DB.IMAGES:
        image_obj = db.images.find_one({'image_urls': image_url})
        if image_obj:
            domain = tldextract.extract(page_url).registered_domain
            collection = get_collection_from_ip_and_domain(ip, domain)
            # IF IMAGE HAS RESULTS FROM THIS IP:
            if has_results_from_collection(image_obj, collection):
                # APPEND URL TO RELEVANT LIST
                ret[image_url] = True
            else:
                ret[image_url] = False
                # GET RESULTS TO THIS GEO
                add_results_from_collection(image_obj, collection)
        else:
            ret[image_url] = False
            start_pipeline.enqueue_call(func=pipeline.start_pipeline,
                                        args=(page_url, image_url),
                                        ttl=2000, result_ttl=2000, timeout=2000)
    return ret


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


def is_image_relevant(image_url, collection_name=None):
    collection_name = collection_name or image_coll_name
    if image_url is not None:
        query = {"image_urls": image_url}
        image_dict = db[collection_name].find_one(query, {'relevant': 1, 'people.items.similar_results': 1})
        if not image_dict:
            return False
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
        res = len(image_dict["people"][0]["items"]) > 0
    except:
        pass
    return res


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
        'people.items.similar_results.': {'$slice': max_results},
        'people.items.similar_results._id': 1,
        'people.items.similar_results.id': 1,
        # 'people.items.svg_url': 1,
        'relevant': 1}

    product_projection = product_projection or {
        #'seeMoreUrl': 1,
        'image.sizes.XLarge.url': 1,
        'images.XLarge': 1,
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
        for item in person["items"]:
            similar_results = []
            for result in item["similar_results"]:
                full_result = collection.find_one({"id": result["id"]}, projection_dict)
                if full_result:
                    # full_result["clickUrl"] = Utils.shorten_url_bitly(full_result["clickUrl"])
                    similar_results.append(full_result)
            item["similar_results"] = similar_results
    return sparse


def image_exists(image_url, collection_name=None):
    collection_name = collection_name or image_coll_name
    image_collection = db[collection_name]
    image_dict = image_collection.find_one({"image_urls": image_url}, {"_id": 1})
    if image_dict is None:
        im_hash = get_hash_of_image_from_url(image_url)
        if im_hash:
            image_dict = image_collection.find_one({"image_hash": im_hash}, {"_id": 1})
    return bool(image_dict)


def merge_items(doc):
    # doc['items'] = []
    # for person in doc['people']:
    #     for item in person['items']:
    #         item['person_bb'] = person['person_bb']
    #         doc['items'].append(item)
    doc['items'] = [item for person in doc['people'] for item in person["items"]]
    del doc["people"]
    return doc


def get_hash_of_image_from_url(image_url):
    if image_url is None:
        logging.warning("Bad image url!")
        return None
    img_arr = Utils.get_cv2_img_array(image_url)
    if img_arr is None:
        logging.warning('couldnt get img_arr from url:' + image_url + ' in get_hash_of_image')
        return None
    return get_hash(img_arr)


def get_hash(image):
    m = hashlib.md5()
    m.update(image)
    url_hash = m.hexdigest()
    return url_hash
