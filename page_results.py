__author__ = 'jeremy'
# MD5 in java http://www.myersdaily.org/joseph/javascript/md5-speed-test-1.html?script=jkm-md5.js#calculations
# after reading a bit i decided not to use named tuple for the image structure
# theirs

import hashlib
import copy
import logging

from bson import objectid
import bson
import pymongo


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


# ours
import Utils
import dbUtils
import background_removal

# similar_results structure - this an example of a similar results answer, with two items
images_entry = \
    {'image_hash': '2403b296b6d0be5e5bb2e74463419b2a',
     'image_urls': ['url1_of_image.jpg', 'url2_of_image.jpg', 'url3_of_image.jpg'],
     'page_urls': ['page1_where_image_appears.html', 'page2_where_image_appears.html',
                   'page3_where_image_appears.html'],
     'relevant': True,  # result of doorman * QC
     'items': [{'category': 'womens-shirt-skirts',
                'svg': 'svg-url',
                'saved_date': 'Jun 23, 1912',
                'similar_items': [{"$ref": 'products', "$id": '<value1>', "$db": 'mydb'},
                                  {"$ref": 'products', "$id": '< value2 >', "$db": 'mydb'},
                                  {"$ref": 'products', "$id": '< value3 >', "$db": 'mydb'}]},

               {'category': 'mens-purse',
                'svg': 'svg-url',
                'saved_date': 'Jun 23, 1912',
                'similar_items': [{"$ref": 'products', "$id": '< value1 >', "$db": 'mydb'},
                                  {"$ref": 'products', "$id": '< value2 >', "$db": 'mydb'},
                                  {"$ref": 'products', "$id": '< value3 >', "$db": 'mydb'}]}]}


# format for results to return to javascript thru web2py . this an example of a similar results answer as returned to web2py
results = {
    "image_hash": "2403b296b6d0be5e5bb2e74463419b2a",
    "image_urls": [
        "url1_of_image.jpg",
        "url2_of_image.jpg",
        "url3_of_image.jpg"
    ],
    "page_urls": [
        "page1_where_image_appears.html",
        "page2_where_image_appears.html",
        "page3_where_image_appears.html"
    ],
    "relevant": "True",
    "people": [
        {
            "face": [
                10,
                20,
                300,
                400
            ],
            "person_id": "bson.ObjectId()",
            "items": [
                {
                    "item_id": '55b89f151f8c82501d18e12f',
                    "category": "womens-shirt-skirts",
                    "svg": "svg-url",
                    "saved_date": "Jun 23, 1912",
                    "similar_items": [
                        {
                            "seeMoreUrl": "url1.html",
                            "image": {
                                "big_ass_dictionary of image info": "the_info"
                            },
                            "LargeImage": "www.largeimg_url1.jpg",
                            "clickUrl": "theurl",
                            "currency": "the_currency",
                            "description": "thedescription",
                            "price": 10.99,
                            "categories": "dict of cats",
                            "pageUrl": "pageUrl",
                            "locale": "US",
                            "name": "itemName",
                            "unbrandedName": "superNameunbranded"
                        },
                        {
                            "seeMoreUrl": "url2.html",
                            "image": {
                                "big_ass_dictionary of image info": "the_info"
                            },
                            "LargeImage": "www.largeimg_url2.jpg",
                            "clickUrl": "theurl",
                            "currency": "the_currency",
                            "description": "thedescription",
                            "price": 10.99,
                            "categories": "dict of cats",
                            "pageUrl": "pageUrl",
                            "locale": "US",
                            "name": "itemName",
                            "unbrandedName": "superNameunbranded"
                        }
                    ]
                },
                {
                    "item_id": '55b89f151f8c82501d18e12e',
                    "category": "awesome_watches",
                    "svg": "svg-url",
                    "saved_date": "Jun 23, 1912",
                    "similar_items": [
                        {
                            "seeMoreUrl": "url.html",
                            "image": {
                                "big_ass_dictionary of image info": "the_info"
                            },
                            "LargeImage": "www.largeimg_url.jpg",
                            "clickUrl": "theurl",
                            "currency": "the_currency",
                            "description": "thedescription",
                            "price": 10.99,
                            "categories": "dict of cats",
                            "pageUrl": "pageUrl",
                            "locale": "US",
                            "name": "itemName",
                            "unbrandedName": "superNameunbranded"
                        },
                        {
                            "seeMoreUrl": "url2.html",
                            "image": {
                                "big_ass_dictionary of image info": "the_info"
                            },
                            "LargeImage": "www.largeimg_url.jpg",
                            "clickUrl": "theurl",
                            "currency": "the_currency",
                            "description": "thedescription",
                            "price": 10.99,
                            "categories": "dict of cats",
                            "pageUrl": "pageUrl",
                            "locale": "US",
                            "name": "itemName",
                            "unbrandedName": "superNameunbranded"
                        }
                    ]
                }
            ]
        },
        {
            "face": [
                10,
                20,
                300,
                400
            ],
            "person_id": "bson.ObjectId()",
            "items": [
                {
                    "item_id": '55b89f151f8c82501d18e12c',
                    "category": "womens-shirt-skirts",
                    "svg": "svg-url",
                    "saved_date": "Jun 23, 1912",
                    "similar_items": [
                        {
                            "seeMoreUrl": "url1.html",
                            "image": {
                                "big_ass_dictionary of image info": "the_info"
                            },
                            "LargeImage": "www.largeimg_url1.jpg",
                            "clickUrl": "theurl",
                            "currency": "the_currency",
                            "description": "thedescription",
                            "price": 10.99,
                            "categories": "dict of cats",
                            "pageUrl": "pageUrl",
                            "locale": "US",
                            "name": "itemName",
                            "unbrandedName": "superNameunbranded"
                        },
                        {
                            "seeMoreUrl": "url2.html",
                            "image": {
                                "big_ass_dictionary of image info": "the_info"
                            },
                            "LargeImage": "www.largeimg_url2.jpg",
                            "clickUrl": "theurl",
                            "currency": "the_currency",
                            "description": "thedescription",
                            "price": 10.99,
                            "categories": "dict of cats",
                            "pageUrl": "pageUrl",
                            "locale": "US",
                            "name": "itemName",
                            "unbrandedName": "superNameunbranded"
                        }
                    ]
                },
                {
                    "item_id": '55b89f151f8c82501d18e12fa',

                    "category": "awesome_watches",
                    "svg": "svg-url",
                    "saved_date": "Jun 23, 1912",
                    "similar_items": [
                        {
                            "seeMoreUrl": "url.html",
                            "image": {
                                "big_ass_dictionary of image info": "the_info"
                            },
                            "LargeImage": "www.largeimg_url.jpg",
                            "clickUrl": "theurl",
                            "currency": "the_currency",
                            "description": "thedescription",
                            "price": 10.99,
                            "categories": "dict of cats",
                            "pageUrl": "pageUrl",
                            "locale": "US",
                            "name": "itemName",
                            "unbrandedName": "superNameunbranded"
                        },
                        {
                            "seeMoreUrl": "url2.html",
                            "image": {
                                "big_ass_dictionary of image info": "the_info"
                            },
                            "LargeImage": "www.largeimg_url.jpg",
                            "clickUrl": "theurl",
                            "currency": "the_currency",
                            "description": "thedescription",
                            "price": 10.99,
                            "categories": "dict of cats",
                            "pageUrl": "pageUrl",
                            "locale": "US",
                            "name": "itemName",
                            "unbrandedName": "superNameunbranded"
                        }
                    ]
                }
            ]
        }
    ]
}

products_db_sample_entry = {
    u'seeMoreUrl': u'http://www.shopstyle.com/browse/womens-tech-accessories/Samsung?pid=uid900-25284470-95',
    u'locale': u'en_US',
    u'image': {u'id': u'4a34ef7c850e64c2435fc3f0a2e0427c', u'sizes': {
        u'XLarge': {u'url': u'https://resources.shopstyle.com/xim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c.jpg',
                    u'width': 328,
                    u'sizeName': u'XLarge', u'height': 410},
        u'IPhoneSmall': {
        u'url': u'https://resources.shopstyle.com/mim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_small.jpg',
        u'width': 100, u'sizeName': u'IPhoneSmall', u'height': 125},
        u'Large': {u'url': u'https://resources.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c.jpg',
                   u'width': 164,
                   u'sizeName': u'Large', u'height': 205},
        u'Medium': {u'url': u'https://resources.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_medium.jpg',
                    u'width': 112, u'sizeName': u'Medium', u'height': 140},
        u'IPhone': {u'url': u'https://resources.shopstyle.com/mim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c.jpg',
                    u'width': 288,
                    u'sizeName': u'IPhone', u'height': 360},
        u'Small': {u'url': u'https://resources.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_small.jpg',
                   u'width': 32, u'sizeName': u'Small', u'height': 40},
        u'Original': {u'url': u'http://bim.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_best.jpg',
                      u'sizeName': u'Original'},
        u'Best': {u'url': u'http://bim.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_best.jpg',
                  u'width': 720,
                  u'sizeName': u'Best', u'height': 900}}},
    u'clickUrl': u'http://api.shopstyle.com/action/apiVisitRetailer?id=468065536&pid=uid900-25284470-95',
    u'retailer': {u'id': u'849', u'name': u'Amazon.com'},
    u'currency': u'USD',
    u'colors': [],
    u'id': 468065536,
    u'badges': [],
    u'extractDate': u'2015-01-12',
    u'alternateImages': [],
    u'archive': True,
    u'dl_version': 0,
    u'preOwned': False,
    u'inStock': True,
    u'brand': {u'id': u'1951', u'name': u'Samsung'},
    u'description': u"Please note: 1> Towallmark is a fashion brand based in China and registered trademark,the only authorized seller of Towallmark branded products.A full line of accessories for all kinds of electronic products,beauty,phone accessories items,clothing,toys,games,home,kitchen and so on. 2> Towallmark provide various kinds of great products at the lowest possible prices to you, welcome to our store and get what you want !!! 3> Towallmark highly appreciate and accept all customers' opinions to improve the selling ,also if anything you unsatisfied, please contact our customer service department for the best solution with any issue.",
    u'seeMoreLabel': u'Samsung Tech Accessories',
    u'price': 5.08,
    u'unbrandedName': u'Towallmark(TM)Flip Leather Case Cover+Bag Straps for Galaxy S4 i9500 Black',
    u'fingerprint': [0.201101154088974, 0.13319680094718933],
    u'rental': False,
    u'categories': [
        {u'shortName': u'Tech', u'localizedId': u'womens-tech-accessories', u'id': u'womens-tech-accessories',
         u'name': u'Tech Accessories'}],
    u'name': u'Towallmark(TM)Flip Leather Case Cover+Bag Straps for Samsung Galaxy S4 i9500 Black',
    u'sizes': [],
    u'lastModified': u'2015-05-21',
    u'brandedName': u'Samsung Towallmark(TM)Flip Leather Case Cover+Bag Straps for Galaxy S4 i9500 Black',
    u'pageUrl': u'http://www.shopstyle.com/p/samsung-towallmark-tm-flip-leather-case-cover-bag-straps-for-galaxy-s4-i9500-black/468065536?pid=uid900-25284470-95',
    u'_id': '557a0a069e31f14ce3901821',
    u'priceLabel': u'$5.08'}



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


def get_hash_of_image_from_url(image_url):
    img_arr = Utils.get_cv2_img_array(image_url)
    if img_arr is None:
        logging.warning('couldnt get img_arr from url:' + image_url + ' in get_hash_of_image')
        return None
    m = hashlib.md5()
    m.update(img_arr)
    url_hash = m.hexdigest()
    logging.debug('url_image hash:' + url_hash + ' for ' + image_url)
    return url_hash

# probably unecesary function, was thinking it would be useful to take different kinds of arguments for some reason
def get_known_similar_results(image_hash=None, image_url=None, page_url=None):
    if image_hash == None and image_url == None and page_url == None:
        logging.warning('get_similar_results wasnt given an id or an image/page url')
        return None

    db = pymongo.MongoClient().mydb
    if image_hash is not None:  #search by imagehash
        query = {'_id': image_hash}
        #query = {"categories": {"$elemMatch": {"image_hash": image_hash}}}
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
    '''

    :param image_url:
    :return: an array of db entries , hopefully the most similar ones to the given image.
    this will require classification (thru qcs ) , fingerprinting, vetting top N items using qc, maybe
    crosschecking, and returning top K results
    '''
    # the goods go here
    # There may be multiple items in an image, so this should return list of items
    # each item having a list of similar results
    # FAKE RESULTS
    logging.debug('starting pipeline')
    db = pymongo.MongoClient().mydb
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
    '''

    :param image_url:
    :return:  should return a human opinion as to whether the image is relevant for us or not
    '''
    # something useful goes here...
    return True


# we wanted to do this as an object , with methods for putting in db
def find_similar_items_and_put_into_db(image_url, page_url):
    '''
        This is for new images - gets the similar items to a given image (at image_url) and puts that the similar item info
        into an images db entry
    :param image_url: url of image to find similar items for, page_url is page it appears on
    :return:  get all the similar items and put them into db if not already there
    uses start_pipeline which is where the actual action is. this just takes results from
    regular db and puts the right fields into the 'images' db
    this does not check if the image already appears elsewhere - whoever called this function
    was supposed to take of that
    '''
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
    db = pymongo.MongoClient().mydb
    db.images.insert(results_dict)
    return results_dict


def update_image_in_db(page_url, image_url, cursor):
    '''
    check each doc in cursor. This is a cursor of docs matching the image at image_url.
    if page_url is there then do nothing, otherwise add page_url to the list page_urls
    :param page_url:
    :param image_url:
    :param cursor:
    :return:
    '''
    db = pymongo.MongoClient().mydb
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
    '''
    this returns all the known similar items for images appearing at page_url (that we know about - maybe images changed since last we checked)
    :type page_url: str
    :return: dictionary of similar results for each image at page
    this should dereference the similar items dbreferences, and return full info except things the front end doesnt care about like fingerprint etc.
    currently the projection operator isnt working for some reason
    there was a bug in dbrefs using projections that was fixed in certain versions , see https://github.com/Automattic/mongoose/issues/1091
    '''
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    if page_url == None:
        logging.warning('results_for_page wasnt given a url')
        return None
    logging.debug('looking for images that appear on page:' + page_url)
    db = pymongo.MongoClient().mydb
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
    if doc == None:
        logging.warning(
            'no image collection entry given for dereferencing (page_results.dereference_image_collection_entry)')
    db = pymongo.MongoClient().mydb
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
        'dl_version': 0,
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
    modified_doc = copy.deepcopy(doc)
    # modified_doc =doc
    logging.debug('trying to dereference ' + str(modified_doc))
    if not 'people' in modified_doc:
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
                    del products_db_entry['dl_version']
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
    '''
    this is for checking a bunch of images on a given page - are they all listed in the images db?
    if not, look for images' similar items and add to images db
    :type page_url: str   This is required in case an image isn;t found in images db
    Then it needs to be matched with similar items and put into db - and the db entry needs the page url
    :type list_of_image_urls: list of the images on a page
    :return:  nothing - just updates db with new images .
    maybe this should return the similar items for each image, tho that can be done by calling results_for_page()
    '''
    if list_of_image_urls == None or page_url == None:
        logging.warning('get_similar_results wasnt given list of image urls and page url')
        return None
    db = pymongo.MongoClient().mydb
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


def get_data_for_specific_image(image_url=None, image_hash=None):
    '''
    this just checks db for an image or hash. It doesn't start the pipeline or update the db
    :param image_url: url of image to find
    :param image_hash: hash (of image) to find
    :return:
    '''
    if image_url == None and image_hash == None:
        logging.warning('page_results.get_data_for_specific_image wasnt given one of image url or image hash')
        return None
    db = pymongo.MongoClient().mydb
    i = 0
    answers = []
    number_found = 0
    number_not_found = 0
    if image_url is not None:
        logging.debug('looking for image ' + image_url + ' in db ')
        query = {"image_urls": image_url}
    else:
        logging.debug('looking for hash ' + image_hash + ' in db ')
        query = {"image_hash": image_hash}
    entry = db.images.find_one(query)
    if entry is not None:
        logging.debug('found image (or hash) in db ')
        # hash gets checked in update_image_in_db(), alternatively it could be checked here
        dereferenced_entry = dereference_image_collection_entry(entry)
        logging.debug('dereferenced entry: ')
        logging.debug(str(dereferenced_entry))
        return dereferenced_entry
    else:
        logging.debug('image / hash  was NOT found in db')
        return None


def remove_one(id):
    db = pymongo.MongoClient().mydb
    db.images.remove({"_id": bson.ObjectId(id)})


def kill_images_collection():
    # connection = Connection('localhost', 27017)  # Connect to mongodb
    #    print(connection.database_names())  # Return a list of db, equal to: > show dbs
    #    db = connection['mydb']  # equal to: > use testdb1
    db = pymongo.MongoClient().mydb
    print(db.collection_names())  # Return a list of collections in 'testdb1'
    print("images exists in db.collection_names()?")  # Check if collection "posts"
    print("images" in db.collection_names())  # Check if collection "posts"
    # exists in db (testdb1
    collection = db['images']
    print('collection.count() = ' + str(collection.count()))  # Check if collection named 'posts' is empty

###DO THIS ONLY IF YOU KNOW WHAT YOU ARE DOING
  #  collection.drop()

###DO THIS TO KILL DB - ASK LIOR AND/OR JEREMY BEFORE DOING THIS

if __name__ == '__main__':
    print('starting')
    # kill_images_collection()
    # remove_one('55b614301f8c8255e7557046')
    #verify_hash_of_image('wefwfwefwe', 'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d.jpg')
    dbUtils.step_thru_images_db(use_visual_output=False, collection='images')
