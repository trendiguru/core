__author__ = 'jeremy'
# MD5 in javascript http://www.myersdaily.org/joseph/javascript/md5-speed-test-1.html?script=jkm-md5.js#calculations
# after reading a bit i decided not to use named tuple for the image structure
# theirs

import hashlib
import logging

# ours
import Utils
import constants

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
db = constants.db
lang = ""
image_coll_name = "images"
prod_coll_name = "products"

'''

def verify_hash_of_image(image_hash, image_url):
    img_arr = Utils.get_cv2_img_array(image_url)
    m = hashlib.md5()
    m.update(img_arr)
    url_hash = m.hexdigest()
    logging.debug('url_image hash:' + url_hash + ' vs image_hash:' + image_hash)
    if url_hash == image_hash:
        return True
    else:
        return False


# probably unneccesary function, was thinking it would be useful to take different kinds of arguments for some reason
def get_known_similar_results(image_hash=None, image_url=None, page_url=None):
    if image_hash is None and image_url is None and page_url is None:
        logging.warning('get_similar_results wasnt given an id or an image/page url')
        return None

    if image_hash is not None:  # search by imagehash
        query = {'_id': image_hash}
        # query = {"categories": {"$elemMatch": {"image_hash": image_hash}}}
        cursor = db.images.find(query)
    #   cursor = db.products.find({'$and': [{"description": {'$regex': keyword}}, query]})

    elif image_url is not None:  # search by image url
        if image_hash is not None:
            if verify_hash_of_image(image_hash, image_url):
                # image has not changed, we can trust a url search
                # query = {'image_urls': {'$elemMatch': {'url': image_url}}}
                query = {'image_urls': image_url}
                cursor = db.images.find(query)
            else:
                # image has changed, so we can't trust url search
                new_images(page_url, image_url)
    else:  # search by page url
        query = {'page_urls': {'$elemMatch': {'url': page_url}}}
    cursor = db.images.find(query)

    n = cursor.count()
    if n == 0:
        # no results for this item were found
        #code to find similar items (eg given image url) could go here
        return None
    elif n > 1:
        logging.warning(str(n) + ' results found')  # maybe only 0 or 1 match should ever be found
    return cursor

def start_pipeline(image_url):
    """

    :param image_url:
    :return: an array of db entries , hopefully the most similar ones to the given image.
    this will require classification (thru qcs ) , fingerprinting, vetting top N items using qc, maybe
    crosschecking, and returning top K results
    """
    # the goods go here
    # There may be multiple items in an image, so this should return list of items
    # each item having a list of similar results
    # FAKE RESULTS
    logging.debug('starting pipeline')
    q = db.products.find()
    similar_1 = q.next()
    similar_2 = q.next()
    similar_3 = q.next()
    similar_4 = q.next()
    # id_1 = objectid.ObjectId(similar_1['_id'])

    id_1 = objectid.ObjectId(similar_1['_id'])
    id_2 = objectid.ObjectId(similar_2['_id'])
    id_3 = objectid.ObjectId(similar_3['_id'])
    id_4 = objectid.ObjectId(similar_4['_id'])
    result = [{"item_id": bson.ObjectId(),
               "category": "womens-skirts",
               "svg": "svg-url",
               "saved_date": "Jun 23, 1912",
               "similar_items":
                   [bson.dbref.DBRef("products", id_1, database="mydb"),
                    bson.dbref.DBRef("products", id_1, database="mydb")]},

              {"item_id": bson.ObjectId(),
               "category": "handbag",
               "svg": "svg-url",
               "saved_date": "Jun 23, 1912",
               "similar_items":
                   [bson.dbref.DBRef("products", id_1, database="mydb"),
                    bson.dbref.DBRef("products", id_1, database="mydb")]}]
    # THIS IS A FAKE PLACEHOLDER RESULT. normally should be an array of products db items

    return result

def qc_assessment_of_relevance(image_url):
    """

    :param image_url:
    :return:  should return a human opinion as to whether the image is relevant for us or not
    """
    # something useful goes here...
    return True

# we wanted to do this as an object , with methods for putting in db
def find_similar_items_and_put_into_db(image_url, page_url):
    """
        This is for new images - gets the similar items to a given image (at image_url) and puts that the similar item info
        into an images db entry
    :param image_url: url of image to find similar items for, page_url is page it appears on
    :return:  get all the similar items and put them into db if not already there
    uses start_pipeline which is where the actual action is. this just takes results from
    regular db and puts the right fields into the 'images' db
    this does not check if the image already appears elsewhere - whoever called this function
    was supposed to take of that
    """
    similar_items_from_products_db = start_pipeline(image_url)  # returns a list of items, each with similar_items
    logging.debug('items returned from pipeline:')
    logging.debug(str(similar_items_from_products_db))

    results_dict = {}
    img_arr = Utils.get_cv2_img_array(image_url)
    if img_arr is None:
        logging.warning('couldnt get image from url:' + str(image_url))
        return None
    m = hashlib.md5()
    m.update(img_arr)
    image_hash = m.hexdigest()
    results_dict["image_hash"] = image_hash
    results_dict["image_urls"] = [image_url]
    results_dict["page_urls"] = [page_url]
    relevance = background_removal.image_is_relevant(img_arr)
    actual_relevance = relevance.is_relevant
    # print('relevance:' + str(actual_relevance))
    relevance = actual_relevance * qc_assessment_of_relevance(image_url)
    results_dict["relevant"] = relevance
    face1 = [311, 47, 44, 44]
    face2 = [399, 30, 45, 45]
    face3 = [116, 15, 47, 47]
    results_dict["people"] = []

    person1 = {'face': face1, 'person_id': bson.ObjectId(), 'items': similar_items_from_products_db}
    person2 = {'face': face2, 'person_id': bson.ObjectId(), 'items': similar_items_from_products_db}
    results_dict["people"].append(person1)
    results_dict["people"].append(person2)
    # results_dict["items"] = similar_items_from_products_db

    #`EDIT FROM HERE        for
    logging.debug('inserting into db:')
    logging.debug(str(results_dict))
    db.images.insert(results_dict)
    return results_dict

def update_image_in_db(page_url, image_url, cursor):
    """
    check each doc in cursor. This is a cursor of docs matching the image at image_url.
    if page_url is there then do nothing, otherwise add page_url to the list page_urls
    :param page_url:
    :param image_url:
    :param cursor:
    :return:
    """
    i = 0
    for doc in cursor:
        # why is there possible more than one doc like this to be updated?  cuz the image
        #may occur on multiple pages
        #       assert(image_url in doc['image_urls']) #the doc should only have been selected from db if the image url mathes
        #acutally thats not nec. true, maybe th hash matched. so forget theassert or assert an 'or'
        i = i + 1

        logging.debug('doc:' + str(doc))
        logging.debug('updating db, doc#' + str(i) + ': looking for page :' + str(page_url) + ' in url list:' + str(
            doc['page_urls']))

        # check if the image in the url is the same one as appears in the supposedly matching doc
        image_hash = doc['image_hash']
        same_image = verify_hash_of_image(image_hash, image_url)
        if not same_image:
            logging.warning(
                'image hash doesnt match the stored hash, maybe the image at ' + str(image_url) + ' changed!')
            continue

        flag = False
        if page_url in doc['page_urls']:
            logging.debug('found url, no need to update, going to next doc')
            continue

        # page url was not found in doc, so add it to list.
        # this is e.g. for the case where a pic appears in two or more pages and this
        # was caught by looking for the image hash  in the images db
        doc['page_urls'].append(page_url)
        id = str(doc['_id'])
        logging.debug('doc before adding page:')
        logging.debug(doc)
        write_result = db.images.update({"_id": objectid.ObjectId(id)}, {"$set": {"page_urls": doc['page_urls']}})
        logging.debug('page_urls changed to:')
        logging.debug(doc)

        # debug - check if doc really changed
        doc = db.images.find_one({"_id": objectid.ObjectId(id)})
        logging.debug('re-found doc:')
        logging.debug(doc)

def get_all_data_for_page(page_url):
    """
    this returns all the known similar items for images appearing at page_url (that we know about - maybe images changed since last we checked)
    :type page_url: str
    :return: dictionary of similar results for each image at page
    this should dereference the similar items dbreferences, and return full info except things the front end doesnt care about like fingerprint etc.
    currently the projection operator isnt working for some reason
    there was a bug in dbrefs using projections that was fixed in certain versions , see https://github.com/Automattic/mongoose/issues/1091
    """
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    if page_url == None:
        logging.warning('results_for_page wasnt given a url')
        return None
    logging.debug('looking for images that appear on page:' + page_url)

    # query = {'page_urls': {'$elemMatch': {'page_url': page_url}}}
    query = {'page_urls': page_url}
    cursor = db.images.find(query)

    n = cursor.count()
    if n == 0:
        # no results for this item were found. maybe return something more informative than 'None'
        logging.debug('no results for pageurl were found in results_for_page')
        return None
    list_of_cursor = list(cursor)
    logging.debug('some images were found in results_for_page search for page url:' + str(page_url))
    logging.debug('the unmodified results as list:' + str(list_of_cursor))

    results = []
    cursor.rewind()
    # This can all be replaced by a call to dereference_image_collection_entry
    for doc in cursor:
        modified_doc = dereference_image_collection_entry(doc)
        results.append(modified_doc)
    logging.debug('the modified results as list:')
    logging.debug(str(results))
    return results

def dereference_image_collection_entry(doc=None):
    if doc is None:
        logging.warning(
            'no image collection entry given for dereferencing (page_results.dereference_image_collection_entry)')

    results = []

    # The projection parameter takes a document of the following form:
    # { field1: <boolean>, field2: <boolean> ... }
    #http://docs.mongodb.org/manual/reference/method/db.collection.find/
    # these are the fields we want to get from the products db
    projection = {
        'seeMoreUrl': 1,
        'locale': 1,
        'image': 1,
        'clickUrl': 1,
        'retailer': 1,
        'currency': 1,
        'colors': 0,
        'id': 0,
        'badges': 0,
        'extractDate': 0,
        'alternateImages': 0,
        'archive': 0,
        'download_data.dl_version': 0,
        'preOwned': 0,
        'inStock': 1,
        'brand': 1,
        'description': 1,
        'seeMoreLabel': 1,
        'price': 1,
        'unbrandedName': 1,
        'fingerprint': 0,
        'rental': 0,
        'categories': 1,
        'name': 1,
        'sizes': 1,
        'lastModified': 0,
        'brandedName': 1,
        'pageUrl': 1,
        '_id': 0,
        'priceLabel': 1}

    # probably necessary since i am fooling with the fields, and the copy is a copy by reference
    modified_doc = doc # copy.deepcopy(doc)
    logging.debug('trying to dereference ' + str(modified_doc))
    if 'people' not in modified_doc:
        logging.debug('no people found in record while trying to dereference ' + str(modified_doc))
        return None
    for person in modified_doc['people']:
        # person['items'] = []
        # orig_items = doc
        new_items = []
        for item in person['items']:  # expand the info in similar_items since in the images db its  just as a reference
            expanded_item = copy.deepcopy(item)
            expanded_item['similar_items'] = []
            for similar_item in item['similar_items']:
                try:
                    # products_db_entry = db.dereference(similar_item, projection)  #runs into type error
                    products_db_entry = db.dereference(similar_item)  # works
                    # below are all the feilds we don't care about and so dont pass on - we have to do this since i can't get the dereference to work
                    del products_db_entry['colors']
                    del products_db_entry['id']
                    del products_db_entry['badges']
                    del products_db_entry['extractDate']
                    del products_db_entry['alternateImages']
                    del products_db_entry['download_data.dl_version']
                    del products_db_entry['preOwned']
                    del products_db_entry['unbrandedName']
                    del products_db_entry['fingerprint']
                    del products_db_entry['rental']
                    del products_db_entry['lastModified']
                    if 'archive' in products_db_entry:
                        del products_db_entry['archive']

                except TypeError:
                    logging.error('dbref is not an instance of DBRef')
                    return None
                except ValueError:
                    logging.error('dbref has a db specified that is different from the current database')
                    return None
                detailed_item = products_db_entry
                large_image = detailed_item['image']['sizes']['Large']['url']
                # add 'large image' so the poor web guy doesnt have too dig that much deeper than he already has to
                detailed_item['LargeImage'] = large_image
                expanded_item['similar_items'].append(detailed_item)
            new_items.append(expanded_item)
        person['items'] = new_items
    logging.debug('the modified results as list:')
    logging.debug(str(modified_doc))
    return modified_doc


def new_images(page_url, list_of_image_urls):
    """
    this is for checking a bunch of images on a given page - are they all listed in the images db?
    if not, look for images' similar items and add to images db
    :type page_url: str   This is required in case an image isn;t found in images db
    Then it needs to be matched with similar items and put into db - and the db entry needs the page url
    :type list_of_image_urls: list of the images on a page
    :return:  nothing - just updates db with new images .
    maybe this should return the similar items for each image, tho that can be done by calling results_for_page()
    :param page_url:
    :param list_of_image_urls:
    :return:
    """

    if list_of_image_urls is None or page_url is None:
        logging.warning('get_similar_results wasnt given list of image urls and page url')
        return None

    i = 0
    answers = []
    number_found = 0
    number_not_found = 0
    for image_url in list_of_image_urls:
        if image_url is None:
            logging.warning('image url #' + str(i) + ' is None')
            continue
        logging.debug('looking for image ' + image_url + ' in db ')
        # query = {"image_urls": {"$elemMatch": {"image_url": image_url}}}
        query = {"image_urls": image_url}
        cursor = db.images.find(query)
        if cursor.count() != 0:
            number_found = number_found + 1
            # answers.append(cursor)
            logging.debug('image ' + image_url + ' was found in db ')
            #hash gets checked in update_image_in_db(), alternatively it could be checked here
            update_image_in_db(page_url, image_url, cursor)
        else:
            number_not_found = number_not_found + 1
            logging.debug('image ' + image_url + ' was NOT found in db, looking for similar items ')
            new_answer = find_similar_items_and_put_into_db(image_url, page_url)
        i = i + 1
    return number_found, number_not_found
'''


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
                                max_results=20, lang=None):
    """
    this just checks db for an image or hash. It doesn't start the pipeline or update the db
    :param image_url: url of image to find
    :param image_hash: hash (of image) to find
    :return:
    """
    if lang:
        set_lang(lang)
    image_collection = db[image_coll_name]

    print "##### image_coll_name: " + image_coll_name + " #####"

    # REMEMBER, image_obj is sparse, similar_results have very few fields.
    image_projection = image_projection or {
        '_id': 1,
        'image_hash': 1,
        'image_urls': 1,
        'page_urls': 1,
        'people.items.category': 1,
        'people.items.category_name': 1,
        'people.items.item_id': 1,
        'people.items.item_idx': 1,
        'people.items.similar_results': {'$slice': max_results},
        'people.items.similar_results._id': 1,
        'people.items.similar_results.id': 1,
        'people.items.svg_url': 1,
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
        full_image_dict = load_similar_results(sparse_image_dict, product_projection)
        merged_dict = merge_items(full_image_dict)
        return merged_dict
    else:
        logging.debug('image / hash  was NOT found in db')
        return None


def load_similar_results(sparse, projection_dict, product_collection_name=None):
    product_collection_name = product_collection_name or prod_coll_name
    collection = db[product_collection_name]
    print "Will load similar results from collection: " + str(collection)
    for person in sparse["people"]:
        for item in person["items"]:
            similar_results = []
            for result in item["similar_results"]:
                full_result = collection.find_one({"id": result["id"]}, projection_dict)
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
    m = hashlib.md5()
    m.update(img_arr)
    url_hash = m.hexdigest()
    logging.debug('url_image hash:' + url_hash + ' for ' + image_url)
    return url_hash


