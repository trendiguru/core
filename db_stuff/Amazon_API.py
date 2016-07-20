'''
useful request parameters:
1. Availability = available
2. Brand
3. BrowseNode - > category id
4. ItemPage
5. Keywords
6. MaximumPrice - 32.42 -> 3242
7. MinimumPrice - same
8. SearchIndex  = localrefernce name


parameters = {
    'AWSAccessKeyId': 'AKIAIQJZVKJKJUUC4ETA',
    'AssociateTag': 'fazz0b-20',
    'Availability': 'Available',
    'Brand': 'Lacoste',
    'Keywords': 'shirts',
    'Operation': 'ItemSearch',
    'SearchIndex': 'FashionWomen',
    'Service': 'AWSECommerceService',
    'Timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    'ResponseGroup': 'ItemAttributes,Offers,Images,Reviews'}


AWSAccessKeyId = AKIAIQJZVKJKJUUC4ETA
AWSAccessKeyPwd = r82svvj4F8h6haYZd3jU+3HkChrW3j8RGcW7WXRK

IMPORTANT!!!
ItemPage can only get numbers from 1 to 10
but each page return 10 results at max
for example, under women's shirts there are 100,000 pages for 1 million items
therefore, we need to divide the requests in a way that less then 100 results will return per search

hierarchy:
7141123011 -> Clothing, Shoes & Jewelry
    7141124011 -> Departments
        7147440011-> Women
            1040660     -> Clothing
                1045024     -> dresses
                    2346727011 -> Casual
                2368343011  -> Tops & Tees
            679337011   -> Shoes


clickUrl -> Item.DetailPageURL

1. query in itemsearch
2. find unique parentASIN
3. use these ParentASIN to do ItemLookup
'''
import argparse

import sys

from Amazon_signature import get_amazon_signed_url
from time import strftime, gmtime, sleep, time
from requests import get
import xmltodict
from db_utils import log2file, print_error, theArchiveDoorman
from ..Yonti import pymongo_utils
from ..constants import db, redis_conn
from rq import Queue
from datetime import datetime
from amazon_worker import insert_items


today_date = str(datetime.date(datetime.now()))

q = Queue('amazon_worker', connection=redis_conn)

blacklist = ['Jewelry', 'Watches', 'Handbags', 'Accessories', 'Lingerie, Sleep & Lounge', 'Socks & Hosiery',
             'Handbags & Wallets', 'Shops', 'Girls', 'Boys', 'Shoes', 'Underwear', 'Baby', 'Sleep & Lounge',
             'Socks', 'Novelty & More', 'Luggage & Travel Gear', 'Uniforms, Work & Safety', 'Costumes & Accessories',
             'Shoe, Jewelry & Watch Accessories', 'Traditional & Cultural Wear', 'Active Underwear', 'Active Socks',
             'Active Supporters', 'Active Base Layers', 'Sports Bras', 'Athletic Socks', 'Athletic Supporters']

base_parameters = {
    'AWSAccessKeyId': 'AKIAIQJZVKJKJUUC4ETA',
    'AssociateTag': 'fazz0b-20',
    'Version': '2013-08-01',
    'Availability': 'Available',
    'Operation': 'ItemSearch',
    'Service': 'AWSECommerceService',
    'Timestamp': strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
    'ResponseGroup': 'ItemAttributes, OfferSummary,Images'}

last_time = time()
log_name = '/home/developer/yonti/amazon_download_stats.log'

colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple', 'magenta', 'cyan', 'grey', 'violet',
          'gold', 'silver', 'khaki', 'turquoise', 'brown']

FashionGender = 'FashionWomen'


def proper_wait(print_flag=False):
    global last_time
    current_time = time()
    time_diff = current_time - last_time
    if time_diff < 1.01:
        sleep(1.01 - time_diff)
        current_time = time()
    if print_flag:
        print ('time diff: %f' % (current_time - last_time))
    last_time=current_time


def truncate_float_to_2_decimal_places(float2round, true_4_str=False):
    float_as_int = int(float2round*100+0.5)
    if true_4_str:
        return str(float_as_int)

    return float_as_int/100.00


def format_price(price_float, period=False):
    """
    input - float
    output - string
    """
    price_str = truncate_float_to_2_decimal_places(price_float, true_4_str=True)

    # verify 4 character string
    while len(price_str) < 4:
        price_str = '0'+price_str

    if period:
        price_str = price_str[:-2]+'.'+price_str[-2:]

    return price_str


def make_itemsearch_request(pagenum, node_id, min_price, max_price, price_flag=True, print_flag=False, color=''):

    parameters = base_parameters.copy()
    parameters['Timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    parameters['SearchIndex'] = FashionGender
    parameters['BrowseNode'] = node_id
    if not price_flag:
        parameters['ResponseGroup'] = 'SearchBins'
    else:
        parameters['ItemPage'] = str(pagenum)
        parameters['MinimumPrice'] = format_price(min_price)
        parameters['MaximumPrice'] = format_price(max_price)
    if len(color):
        parameters['Keywords'] = color

    req = get_amazon_signed_url(parameters, 'GET', False)
    proper_wait()
    res = get(req)
    # last_time = time()

    if res.status_code != 200:
        if print_flag:
            print_error('Bad request', req)
        return [], -1

    res_dict = dict(xmltodict.parse(res.text))
    if 'ItemSearchResponse' not in res_dict.keys():
        if print_flag:
            print_error('No ItemSearchResponse', req)
        return [], -2

    res_dict = dict(res_dict['ItemSearchResponse']['Items'])
    if 'TotalResults' in res_dict.keys():
        results_count = int(res_dict['TotalResults'])
    else:
        if print_flag:
            print_error('bad query', req)
        return [], -3

    if results_count == 0:
        if print_flag:
            price_range = 'PriceRange: %s -> %s ' % (format_price(min_price, True), format_price(max_price, True))
            print_error('no results for %s' % price_range)
        return [], -4

    if 'TotalPages' not in res_dict.keys():
        if print_flag:
            print_error('no TotalPages in dict keys')
        return [], -5

    if 'Errors' in res_dict.keys():
        if print_flag:
            print_error('Error', req)
        return [], -6

    return res_dict, results_count


def process_results(collection_name, pagenum, node_id, min_price, max_price, family_tree, res_dict=None,
                    items_in_page=10, print_flag=False, color=''):
    if pagenum is not 1:
        res_dict, new_item_count = make_itemsearch_request(pagenum, node_id, min_price, max_price,
                                                           print_flag=print_flag, color=color)
        if new_item_count < 2:
            return 0

    item_list = res_dict['Item']
    item_count = len(item_list)
    q.enqueue(insert_items, args=(collection_name, item_list, items_in_page, print_flag, family_tree), timeout=5400)

    return item_count


def filter_by_color(collection_name, node_id, price, family_tree):
    for color in colors:
        res_dict, results_count = make_itemsearch_request(1, node_id, price, price, color=color)
        if results_count < 2:
            summary = 'Name: %s, PriceRange: %s -> %s , ResultCount: %d (color -> %s)' \
                      % (family_tree, format_price(price, True), format_price(price, True), results_count, color)
            log2file(mode='a', log_filename=log_name, message=summary)
            continue

        new_items_count=0
        total_pages = int(res_dict['TotalPages'])
        if total_pages == 1:
            num_of_items_in_page = results_count
        else:
            num_of_items_in_page = 10
        new_items_count += process_results(collection_name, 1, node_id, price, price, family_tree=family_tree,
                                           res_dict=res_dict, items_in_page=num_of_items_in_page,color=color)

        for pagenum in range(2, total_pages + 1):
            if pagenum == total_pages:
                num_of_items_in_page = results_count - 10 * (pagenum - 1)
                if num_of_items_in_page < 2:
                    break
            new_items_count += process_results(collection_name, pagenum, node_id, price, price, family_tree=family_tree,
                                               items_in_page=num_of_items_in_page, color=color)

        summary = 'Name: %s, PriceRange: %s -> %s , ResultCount: %d (color -> %s)' \
                  % (family_tree, format_price(price, True), format_price(price, True), results_count, color)
        log2file(mode='a', log_filename=log_name, message=summary)
    return 0


def get_results(node_id, collection_name='moshe',  price_flag=True, max_price=3000.0, min_price=0.0, results_count_only=False,
                family_tree='moshe'):
    if not results_count_only:
        cache_name = collection_name+'_cache'
        collection_cache = db[cache_name]
        collection_cache.update_one({'node_id':node_id}, {'$set':{'last_max':max_price}})

    res_dict, results_count = make_itemsearch_request(1, node_id, min_price, max_price, price_flag=price_flag)
    if results_count < 2:
        summary = 'Name: %s, PriceRange: %s -> %s , ResultCount: %d' \
                  % (family_tree, format_price(min_price, True), format_price(max_price, True), results_count)
        log2file(mode='a', log_filename=log_name, message=summary)
        return 0

    if results_count_only:
        return results_count

    new_items_count = 0
    total_pages = int(res_dict['TotalPages'])
    color_flag = False
    if results_count > 100:
        # print ('min : %.4f -> max : %.4f' %(min_price, max_price))
        diff = truncate_float_to_2_decimal_places(max_price-min_price)
        if diff <= 0.01:
            color_flag = True
            total_pages = 10
        elif diff <= 0.02:
            new_items_count += get_results(node_id, collection_name,
                                           min_price=max_price, max_price=max_price, family_tree=family_tree)
            new_min_price = max_price - 0.01
            new_items_count += get_results(node_id, collection_name,
                                           min_price=new_min_price, max_price=new_min_price, family_tree=family_tree)
            return new_items_count
        else:
            mid_price = (max_price+min_price)/2.0
            mid_price_rounded = truncate_float_to_2_decimal_places(mid_price)
            new_items_count += get_results(node_id, collection_name,
                                           min_price=mid_price_rounded, max_price=max_price, family_tree=family_tree)
            new_items_count += get_results(node_id, collection_name,
                                           min_price=min_price, max_price=mid_price_rounded-0.01,
                                           family_tree=family_tree)
            return new_items_count

    if total_pages == 1:
        num_of_items_in_page = results_count
    else:
        num_of_items_in_page = 10
    new_items_count += process_results(collection_name, 1, node_id, min_price, max_price, family_tree=family_tree,
                                       res_dict=res_dict, items_in_page=num_of_items_in_page)

    for pagenum in range(2, total_pages+1):
        if pagenum == total_pages:
            num_of_items_in_page = results_count-10*(pagenum-1)
            if num_of_items_in_page < 2:
                break
        new_items_count += process_results(collection_name, pagenum, node_id, min_price, max_price,
                                           family_tree=family_tree, items_in_page=num_of_items_in_page)
    summary = 'Name: %s, PriceRange: %s -> %s , ResultCount: %d ' \
              % (family_tree, format_price(min_price, True), format_price(max_price, True), results_count)
    log2file(mode='a', log_filename=log_name, message=summary)
    if color_flag:
        filter_by_color(collection_name, node_id, max_price, family_tree=family_tree)
        max_rounded = format_price(max_price)
        min_rounded = format_price(min_price)
        if max_rounded != min_rounded:
            filter_by_color(collection_name, node_id, min_price, family_tree=family_tree)
    return new_items_count


def build_category_tree(root='7141124011', tab=0, parents=[], delete_collection=True):

    if delete_collection:
        db.amazon_category_tree.delete_many({})

    parameters = base_parameters.copy()
    parameters['Timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    parameters['Operation'] = 'BrowseNodeLookup'
    parameters['ResponseGroup'] = 'BrowseNodeInfo'
    parameters['BrowseNodeId'] = root
    req = get_amazon_signed_url(parameters, 'GET', False)
    res = get(req)

    if res.status_code != 200 :
        print_error('Bad request', req)
        return None

    res_dict = dict(xmltodict.parse(res.text))
    if 'BrowseNodeLookupResponse' not in res_dict.keys():
        print_error('No BrowseNodeLookupResponse', req)
        return None

    res_dict = dict(res_dict['BrowseNodeLookupResponse']['BrowseNodes']['BrowseNode'])
    if 'Children' in res_dict.keys():
        children = res_dict['Children']['BrowseNode']
    else:
        children = []

    name = res_dict['Name']
    if name in blacklist:
        return name

    node_id = res_dict['BrowseNodeId']
    result_count = get_results(node_id, price_flag=False, results_count_only=True)

    leaf = {'Name': name,
            'BrowseNodeId': node_id,
            'Parents': parents,
            'Children': {'count': len(children),
                         'names': []},
            'TotalResults': result_count}

    tab_space = '\t' * tab
    print('%sName: %s,  NodeId: %s,  Children: %d , result_count: %d'
          % (tab_space, name, leaf['BrowseNodeId'], leaf['Children']['count'], result_count))

    tab += 1
    if len(parents) == 0:
        p = [name]
    else:
        p = [x for x in parents]
        p.append(name)

    for child in children:
        sleep(1.01)
        if 'BrowseNodeId' not in child.keys():
            continue
        child_id = child['BrowseNodeId']
        child_name = build_category_tree(child_id, tab, p)
        if child_name is None:  # try again
            print ('##################################################################################################')
            child_name = build_category_tree(child_id, tab,  p)

        leaf['Children']['names'].append((child_id,child_name))

    db.amazon_category_tree.delete_one({'BrowseNodeId': node_id})
    db.amazon_category_tree.insert_one(leaf)
    return name


def clear_duplicates(name):
    collection = db[name]
    before = collection.count()
    all_items = collection.find()
    for item in all_items:
        item_id = item['_id']
        asin= item['asin']
        asin_exists = collection.find({'asin':asin})
        if asin_exists.count()>1:
            id_to_del = []
            for tmp_item in asin_exists:
                tmp_id = tmp_item['_id']
                if tmp_id == item_id:
                    continue
                id_to_del.append(tmp_id)
            if len(id_to_del):
                collection.delete_many({'_id':{'$in':id_to_del}})

        parent = item['parent_asin']
        parent_exists = collection.find({'parent_asin': parent})
        if parent_exists.count() > 1:
            id_to_del = []
            current_sizes = item['sizes']
            current_color = item['color']
            for tmp_item in parent_exists:
                tmp_id = tmp_item['_id']
                tmp_color = tmp_item['color']
                if tmp_id == item_id or tmp_color != current_color:
                    continue
                tmp_sizes = tmp_item['sizes']
                for size in tmp_sizes:
                    if size not in current_sizes:
                        current_sizes.append(size)
                id_to_del.append(tmp_id)
            if len(id_to_del):
                collection.delete_many({'_id': {'$in': id_to_del}})

        img_hash = item['img_hash']
        hash_exists = collection.find({'img_hash': img_hash})
        if hash_exists.count() > 1:
            id_to_del = []
            for tmp_item in hash_exists:
                tmp_id = tmp_item['_id']
                if tmp_id == item_id:
                    continue
                id_to_del.append(tmp_id)
            if len(id_to_del):
                collection.delete_many({'_id': {'$in': id_to_del}})

        img_url = item['images']['XLarge']
        img_url_exists = collection.find({'images.XLarge': img_url})
        if img_url_exists.count() > 1:
            id_to_del = []
            for tmp_item in img_url_exists:
                tmp_id = tmp_item['_id']
                if tmp_id == item_id:
                    continue
                id_to_del.append(tmp_id)
            if len(id_to_del):
                collection.delete_many({'_id': {'$in': id_to_del}})

    print_error('clear duplicates', 'count before : %d\ncount after : %d' % (before, collection.count()))


def download_all(country_code='US', gender='Female', del_collection=False, del_cache=False, cat_tree=False):

    collection_name = 'amazon_%s_%s' % (country_code, gender)
    cache_name = collection_name+'_cache'
    collection_cache = db[cache_name]
    if cat_tree:
        build_category_tree()

    if del_collection:
        pymongo_utils.delete_or_and_index(collection_name, ['id', 'img_hash', 'categories', 'images.XLarge',
                                                            'download_data.dl_version', 'asin', 'parent_asin',
                                                            'features.color'], delete_flag=True)

    if del_cache:
        pymongo_utils.delete_or_and_index(cache_name, ['node_id'], delete_flag=True)

    if gender is 'Female':
        parent_gender = 'Women'
    else:
        parent_gender = 'Men'

    leafs_cursor = db.amazon_category_tree.find({'Children.count': 0, 'Parents': parent_gender})
    leafs = [x for x in leafs_cursor]
    iteration = 0
    while len(leafs):
        if iteration > 5:
            break
        not_finished = []

        for leaf in leafs:
            leaf_name = '->'.join(leaf['Parents']) + '->' + leaf['Name']
            node_id = leaf['BrowseNodeId']
            cache_exists = collection_cache.find_one({'node_id': node_id})
            max_price = 3000.0
            if cache_exists:
                if cache_exists['last_max'] > 0.00:
                    max_price = cache_exists['last_max']
                    print ('node id: %s didn\'t finish -> continuing from %.2f' % (node_id, max_price))
                else:
                    print ('node id: %s already downloaded!' % node_id)
                    continue
            else:
                cache = {'node_id': node_id,
                         'item_count': 0,
                         'last_max': max_price}
                collection_cache.insert_one(cache)

            try:
                new_items_count = get_results(node_id, collection_name, max_price=max_price, results_count_only=False,
                                              family_tree=leaf_name)
                print('node id: %s download done -> %d new_items downloaded' % (node_id, new_items_count))
                collection_cache.update_one({'node_id': node_id},
                                            {'$set': {'item_count': new_items_count, 'last_max': 0.00}})

            except Exception as e:
                print_error('ERROR', 'node id: %s failed!\n %s' % (node_id, e))
                not_finished.append(leaf)
        leafs = not_finished
        do_again = len(leafs)
        if do_again:
            print_error('%d leafs to do again!' % do_again)
        iteration += 1

    clear_duplicates(collection_name)
    theArchiveDoorman(collection_name, instock_limit=7, archive_limit=14)
    collection_cache.delete_many({})
    message = 'amazon %s %s is Done!' % (country_code, gender)
    log2file(mode='a', log_filename=log_name, message=message, print_flag=True)


def getUserInput():
    parser = argparse.ArgumentParser(description='"@@@ Amazon Download @@@')
    parser.add_argument('-c', '--code', default="US", dest="country_code",
                        help='country code - currently doing only US')
    parser.add_argument('-g', '--gender', dest="gender",
                        help='specify which gender to download', required=True)
    parser.add_argument('-d', '--delete', dest="delete_all", default=False, action='store_true',
                        help='delete all items in collection')
    parser.add_argument('-f', '--fresh', dest="delete_cache", default=False, action='store_true',
                        help='delete all cache and start a fresh download')
    parser.add_argument('-t', '--tree', dest="tree", default=False, action='store_true',
                        help='build category tree from scratch')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    user_input = getUserInput()
    c_c = user_input.country_code
    col_gender = user_input.gender
    gender_upper = col_gender.upper()
    if gender_upper == 'FEMALE':
        col_gender= 'Female'
    elif gender_upper == 'MALE':
        global FashionGender
        FashionGender='FashionMen'
        col_gender= 'Male'
    else:
        print("bad input - gender should be only Female or Male ")
        sys.exit(1)

    cc_upper = c_c.upper()
    if cc_upper != 'US':
        print("bad input - for now only working on US")
        sys.exit(1)

    col_name = 'amazon_%s_%s' % (cc_upper, col_gender)
    title = "@@@ amazon Download @@@"
    title2 = "you choose to update the %s collection" % col_name
    log2file(mode='w', log_filename=log_name, message=title, print_flag=True)
    log2file(mode='a', log_filename=log_name, message=title2, print_flag=True)

    delete_all = user_input.delete_all
    if delete_all:
        warning = 'you choose to delete all items!!!'
        sure = 'are you sure? (yes/no)'
        print_error(warning)
        ans = raw_input(sure)
        if ans != 'yes':
            warning = 'you choose to continue WITHOUT deleting'
            delete_all = False
        else:
            warning = 'you choose to DELETE all'
        print_error(warning)

    delete_cache = user_input.delete_cache
    build_tree = user_input.tree
    if build_tree:
        delete_cache = True
    download_all(country_code=cc_upper, gender=col_gender, del_collection=delete_all, del_cache=delete_cache,
                 cat_tree=build_tree)

    # forest_job = forest.enqueue(plantForests4AllCategories, col_name=col, timeout=3600)
    # while not forest_job.is_finished and not forest_job.is_failed:
    #     time.sleep(300)
    # if forest_job.is_failed:
    #     print ('annoy plant forest failed')

