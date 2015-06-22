__author__ = 'jeremy'
# MD5 in java http://www.myersdaily.org/joseph/javascript/md5-speed-test-1.html?script=jkm-md5.js#calculations
# after reading a bit i decided not to use named tuple for the image structure
# theirs
import logging
import hashlib

import pymongo


# ours
import Utils
import background_removal

# similar_results structure - this an example of a similar results answer, with two items
similar_results_dict = {'image_hash': 'md5hash of image, like 2403b296b6d0be5e5bb2e74463419b2a',
                        'image_urls': [{'image_url': 'image_url1_where_image_appears.jpg'},
                                       {'image_url': 'image_url2_where_image_appears.jpg'},
                                       {'image_url': 'image_url3_where_image_appears.jpg'}],
                        'page_urls': [{'page_url': 'pageurl1_where_image_appears.html'},
                                      {'page_url': 'pageurl2_where_image_appears.html'},
                                      {'page_url': 'pageurl3_where_image_appears.html'}],
                        # this lameness (dict instead of flat array) is apparently necessary to use $elemmatch
                        'relevant': True,  #result of doorman * QC
                        'results': [{'category': 'womens-shirt-skirts',
                                     'svg': 'svg-url',
                                     'similar_items': [{
                                                           'seeMoreUrl': 'http://www.shopstyle.com/browse/womens-tech-accessories/Suunto?pid=uid900-25284470-95',
                                                           'image': {u'id': u'c8af6068982f408205491817fe4cad5d',
                                                                     u'sizes': {
                                                                         u'XLarge': {
                                                                             u'url': u'http://resources.shopstyle.com/xim/c8/af/c8af6068982f408205491817fe4cad5d.jpg',
                                                                             u'width': 328, u'sizeName': u'XLarge',
                                                                             u'height': 410},
                                                                         u'IPhoneSmall': {
                                                                             u'url': u'http://resources.shopstyle.com/mim/c8/af/c8af6068982f408205491817fe4cad5d_small.jpg',
                                                                             u'width': 100, u'sizeName': u'IPhoneSmall',
                                                                             u'height': 125},
                                                                         u'Large': {
                                                                             u'url': u'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d.jpg',
                                                                             u'width': 164, u'sizeName': u'Large',
                                                                             u'height': 205},
                                                                         u'Medium': {
                                                                             u'url': u'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d_medium.jpg',
                                                                             u'width': 112, u'sizeName': u'Medium',
                                                                             u'height': 140},
                                                                         u'IPhone': {
                                                                             u'url': u'http://resources.shopstyle.com/mim/c8/af/c8af6068982f408205491817fe4cad5d.jpg',
                                                                             u'width': 288, u'sizeName': u'IPhone',
                                                                             u'height': 360},
                                                                         u'Small': {
                                                                             u'url': u'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d_small.jpg',
                                                                             u'width': 32, u'sizeName': u'Small',
                                                                             u'height': 40},
                                                                         u'Original': {
                                                                             u'url': u'http://bim.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d_best.jpg',
                                                                             u'sizeName': u'Original'}, u'Best': {
                                                                         u'url': u'http://bim.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d_best.jpg',
                                                                         u'width': 720, u'sizeName': u'Best',
                                                                         u'height': 900}}},
                                                           'LargeImage': {
                                                               u'url': u'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d.jpg',
                                                               u'width': 164, u'sizeName': u'Large', u'height': 205},
                                                           'clickUrl': 'http://api.shopstyle.com/action/apiVisitRetailer?id=477060145&pid=uid900-25284470-95',
                                                           'currency': 'USD',
                                                           'description': 'SUUNTO Hi-tech Accessories. The famous Suunto Core packs easy to use outdoor functi blabla',
                                                           'extractDate': '2015-04-24',
                                                           'price': 594.0,
                                                           'categories': [{u'shortName': u'Tech',
                                                                           u'localizedId': u'womens-tech-accessories',
                                                                           u'id': u'womens-tech-accessories',
                                                                           u'name': u'Tech Accessories'}],
                                                           'pageUrl': 'http://www.shopstyle.com/p/suunto-hi-tech-accessories/477060145?pid=uid900-25284470-95',
                                                           'locale': 'En_US',
                                                           'name': 'SUUNTO Hi-tech Accessories',
                                                           'unbrandedName': 'Hi-tech Accessories'},

                                                       {
                                                           'seeMoreUrl': 'http://www.shopstyle.com/browse/womens-tech-accessories/Roccobarocco?pid=uid900-25284470-95',
                                                           'image': {u'id': u'994dfb526bc7b22aa59fc86f4f314583',
                                                                     u'sizes': {
                                                                         u'XLarge': {
                                                                             u'url': u'http://resources.shopstyle.com/xim/99/4d/994dfb526bc7b22aa59fc86f4f314583.jpg',
                                                                             u'width': 328, u'sizeName': u'XLarge',
                                                                             u'height': 410},
                                                                         u'IPhoneSmall': {
                                                                             u'url': u'http://resources.shopstyle.com/mim/99/4d/994dfb526bc7b22aa59fc86f4f314583_small.jpg',
                                                                             u'width': 100, u'sizeName': u'IPhoneSmall',
                                                                             u'height': 125},
                                                                         u'Large': {
                                                                             u'url': u'http://resources.shopstyle.com/pim/99/4d/994dfb526bc7b22aa59fc86f4f314583.jpg',
                                                                             u'width': 164, u'sizeName': u'Large',
                                                                             u'height': 205},
                                                                         u'Medium': {
                                                                             u'url': u'http://resources.shopstyle.com/pim/99/4d/994dfb526bc7b22aa59fc86f4f314583_medium.jpg',
                                                                             u'width': 112, u'sizeName': u'Medium',
                                                                             u'height': 140},
                                                                         u'IPhone': {
                                                                             u'url': u'http://resources.shopstyle.com/mim/99/4d/994dfb526bc7b22aa59fc86f4f314583.jpg',
                                                                             u'width': 288, u'sizeName': u'IPhone',
                                                                             u'height': 360},
                                                                         u'Small': {
                                                                             u'url': u'http://resources.shopstyle.com/pim/99/4d/994dfb526bc7b22aa59fc86f4f314583_small.jpg',
                                                                             u'width': 32, u'sizeName': u'Small',
                                                                             u'height': 40},
                                                                         u'Original': {
                                                                             u'url': u'http://bim.shopstyle.com/pim/99/4d/994dfb526bc7b22aa59fc86f4f314583_best.jpg',
                                                                             u'sizeName': u'Original'}, u'Best': {
                                                                         u'url': u'http://bim.shopstyle.com/pim/99/4d/994dfb526bc7b22aa59fc86f4f314583_best.jpg',
                                                                         u'width': 720, u'sizeName': u'Best',
                                                                         u'height': 900}}},
                                                           'LargeImage': {
                                                               u'url': u'http://resources.shopstyle.com/pim/99/4d/994dfb526bc7b22aa59fc86f4f314583.jpg',
                                                               u'width': 164, u'sizeName': u'Large', u'height': 205},
                                                           'clickUrl': 'http://api.shopstyle.com/action/apiVisitRetailer?id=471078722&pid=uid900-25284470-95',
                                                           'currency': 'USD',
                                                           'description': 'ROCCOBAROCCO Hi-tech Accessories. varnished effect, logo detail, solid color. 100% Polyurethane',
                                                           'extractDate': '2015-04-24',
                                                           'price': 94.0,
                                                           'categories': [{u'shortName': u'Tech',
                                                                           u'localizedId': u'womens-tech-accessories',
                                                                           u'id': u'womens-tech-accessories',
                                                                           u'name': u'Tech Accessories'}],
                                                           'pageUrl': 'http://www.shopstyle.com/p/roccobarocco-hi-tech-accessories/471078722?pid=uid900-25284470-95',
                                                           'locale': 'En_US',
                                                           'name': 'SUUNTO Hi-tech Accessories',
                                                           'unbrandedName': 'Hi-tech Accessories'}
                                     ]}]}

products_db_sample_entry = {
u'seeMoreUrl': u'http://www.shopstyle.com/browse/womens-tech-accessories/Samsung?pid=uid900-25284470-95',
u'locale': u'en_US',
u'image': {u'id': u'4a34ef7c850e64c2435fc3f0a2e0427c', u'sizes': {
u'XLarge': {u'url': u'https://resources.shopstyle.com/xim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c.jpg', u'width': 328,
            u'sizeName': u'XLarge', u'height': 410},
u'IPhoneSmall': {u'url': u'https://resources.shopstyle.com/mim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_small.jpg',
                 u'width': 100, u'sizeName': u'IPhoneSmall', u'height': 125},
u'Large': {u'url': u'https://resources.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c.jpg', u'width': 164,
           u'sizeName': u'Large', u'height': 205},
u'Medium': {u'url': u'https://resources.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_medium.jpg',
            u'width': 112, u'sizeName': u'Medium', u'height': 140},
u'IPhone': {u'url': u'https://resources.shopstyle.com/mim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c.jpg', u'width': 288,
            u'sizeName': u'IPhone', u'height': 360},
u'Small': {u'url': u'https://resources.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_small.jpg',
           u'width': 32, u'sizeName': u'Small', u'height': 40},
u'Original': {u'url': u'http://bim.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_best.jpg',
              u'sizeName': u'Original'},
u'Best': {u'url': u'http://bim.shopstyle.com/pim/4a/34/4a34ef7c850e64c2435fc3f0a2e0427c_best.jpg', u'width': 720,
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
u'fingerprint': [0.201101154088974, 0.13319680094718933, 0.055736903101205826, 0.5849725008010864, 0.6874059438705444,
                 0.9091284275054932, 0.12757940590381622, 0.3364734649658203, 0.2218434065580368, 0.04536754637956619,
                 0.01109516154974699, 0.0018541779136285186, 0.0002990609500557184, 2.9906095733167604e-05,
                 8.9718283561524e-05, 5.981219146633521e-05, 0.0008373706368729472, 0.0, 5.981219146633521e-05,
                 5.981219146633521e-05, 0.0, 0.000179436567123048, 0.003349482547491789, 0.0002691548434086144,
                 0.002003708388656378, 0.01925952546298504, 0.0007476523751392961, 0.012052156031131744,
                 0.11163945496082306, 0.023117411881685257, 0.08173336088657379, 0.20859500765800476,
                 0.19929420948028564, 0.13915306329727173, 0.13021112978458405, 0.08457443863153458,
                 0.05906453728675842, 0.0345415398478508, 0.027244452387094498, 0.020276332274079323,
                 0.014953047037124634, 0.014504455961287022, 0.009599856100976467, 0.007087744306772947,
                 0.0051139420829713345, 0.003588731400668621, 0.00430647749453783, 0.0030803277622908354,
                 0.002661642385646701, 0.0017644596518948674, 0.0009569950634613633, 0.0019139901269227266,
                 0.0011663377517834306, 0.0007476523751392961, 0.000358873134246096, 8.9718283561524e-05],
u'rental': False,
u'categories': [{u'shortName': u'Tech', u'localizedId': u'womens-tech-accessories', u'id': u'womens-tech-accessories',
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
    print('url_image hash:' + url_hash + ' vs image_hash:' + image_hash)
    if url_hash == image_hash:
        return True
    else:
        return False


# probably unecesary function, was thinking it would be useful to take different kinds of arguments for some reason
def get_known_similar_results(image_hash=None, image_url=None, page_url=None):
    if image_hash == None and image_url == None and page_url == None:
        logging.warning('get_similar_results wasnt given an id or an image/page url')
        return None

    db = pymongo.MongoClient().mydb
    if image_hash is not None:  #search by imagehash
        query = {'image_hash': image_hash}
        #query = {"categories": {"$elemMatch": {"image_hash": image_hash}}}
        cursor = db.images.find(query)
    #   cursor = db.products.find({'$and': [{"description": {'$regex': keyword}}, query]})

    elif image_url is not None:  # search by image url
        if image_hash is not None:
            if verify_hash_of_image(image_hash, image_url):
                # image has not changed, we can trust a url search
                query = {'image_urls': {'$elemMatch': {'url': image_url}}}
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


def results_for_page(page_url):
    if page_url == None:
        logging.warning('results_for_page wasnt given a url')
        return None
    print('looking for images that appear on page:' + page_url)
    db = pymongo.MongoClient().mydb
    query = {'page_urls': {'$elemMatch': {'page_url': page_url}}}
    cursor = db.images.find(query)

    n = cursor.count()
    if n == 0:
        # no results for this item were found
        # code to find similar items (eg given image url) could go here
        return None

    return list(cursor)



def start_pipeline(image_url):
    '''

    :param image_url:
    :return: an array of db entries , hopefully the most similar ones to the given image.
    this will require classification (thru qcs ) , fingerprinting, vetting top N items using qc, maybe
    crosschecking, and returning top K results
    '''
    # the goods go here

    # THIS IS A FAKE PLACEHOLDER RESULT. normally should be an array of products db items
    first_result = products_db_sample_entry
    db = pymongo.MongoClient().mydb
    second_result = db.products.find_one()

    return [first_result, second_result]


def qc_analysis_of_relevance(image_url):
    '''

    :param image_url:
    :return:  should return a human opinion as to whether the image is relevant for us or not
    '''
    # something useful goes here...
    return True


def find_similar_items_and_put_into_db(image_url, page_url):
    '''

    :param image_url: url of image to find similar items for
    :return:  get all the similar items and put them into db if not already there
    uses start_pipeline which is where the actual action is. this just takes results from
    regular db and puts the right fields into the 'images' db
    '''
    similar_items_from_db = start_pipeline(image_url)  # this will be a list of regular db entries
    print('similar items:')
    print similar_items_from_db
    # this function is supposed to put these products db entries into the images db format
    results_dict = {}
    img_arr = Utils.get_cv2_img_array(image_url)
    if img_arr is None:
        logging.warning('couldnt get image')
        return None
    m = hashlib.md5()
    m.update(img_arr)
    image_hash = m.hexdigest()
    results_dict['image_hash'] = image_hash
    results_dict['image_urls'] = [{'image_url': image_url}]
    results_dict['page_urls'] = [{'page_url': page_url}]
    relevance = background_removal.image_is_relevant(img_arr)
    relevance = relevance * qc_analysis_of_relevance(image_url)
    results_dict['relevant'] = relevance
    similar_items = []
    for similar_item in similar_items_from_db:
        entry = {}
        entry['seeMoreUrl'] = similar_item['seeMoreUrl']
        entry['image'] = similar_item['image']
        entry['LargeImage'] = similar_item['image']['sizes']['Large']
        entry['clickUrl'] = similar_item['clickUrl']
        entry['currency'] = similar_item['currency']
        entry['description'] = similar_item['description']
        entry['price'] = similar_item['price']
        entry['categories'] = similar_item['categories']
        entry['pageUrl'] = similar_item['pageUrl']
        entry['locale'] = similar_item['locale']
        entry['name'] = similar_item['name']
        entry['unbrandedName'] = similar_item['unbrandedName']
        similar_items.append(entry)
    results_dict['similar_items'] = similar_items
    db = pymongo.MongoClient().mydb
    db.images.insert(results_dict)
    print('inserted into db:')
    print(results_dict)
    return results_dict


def update_images_db(image_url, page_url, new_answer):
    pass


def new_images(page_url, list_of_image_urls):
    if list_of_image_urls == None or page_url == None:
        logging.warning('get_similar_results wasnt given list of image urls and page url')
        return None

    db = pymongo.MongoClient().mydb
    i = 0
    answers = []
    for image_url in list_of_image_urls:
        if image_url is None:
            logging.warning('image url #' + str(i) + ' is None')
            continue
        query = {'image_url': image_url}
        cursor = db.images.find(query)
        if cursor.count() != 0:
            answers.append(cursor)
        else:
            new_answer = find_similar_items_and_put_into_db(image_url, page_url)
            answers.append(new_answer)
            update_images_db(image_url, page_url, new_answer)
        i = i + 1

    return answers

if __name__ == '__main__':
    print('starting')
    verify_hash_of_image('wefwfwefwe', 'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d.jpg')