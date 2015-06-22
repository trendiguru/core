__author__ = 'jeremy'
# MD5 in java http://www.myersdaily.org/joseph/javascript/md5-speed-test-1.html?script=jkm-md5.js#calculations
# after reading a bit i decided not to use named tuple for the image structure

import logging

import pymongo


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


def get_similar_results(image_hash=None, image_url=None, page_url=None):
    if image_hash == None and image_url == None and page_url == None:
        logging.warning('get_similar_results wasnt given an id or an image/page url')
        return None

    db = pymongo.MongoClient().images
    if image_hash is not None:  #search by imagehash
        query = {'image_hash': image_hash}
        #query = {"categories": {"$elemMatch": {"image_hash": image_hash}}}
        cursor = db.products.find(query)
    #   cursor = db.products.find({'$and': [{"description": {'$regex': keyword}}, query]})

    elif image_url is not None:  # search by image url
        query = {'image_urls': {'$elemMatch': {'url': image_url}}}
        cursor = db.products.find(query)

    else:  # search by page url
        query = {'page_urls': {'$elemMatch': {'url': page_url}}}
    cursor = db.products.find(query)

    n = cursor.count()
    if n == 0:
        return None
    elif n > 1:
        logging.warning(str(n) + ' results found')  # maybe only 0 or 1 match should ever be found
    return cursor


def new_image(image_hash=None, image_url=None, page_url=None):
    if image_hash == None and image_url == None and page_url == None:
        logging.warning('get_similar_results wasnt given an id or an image/page url')
        return None

