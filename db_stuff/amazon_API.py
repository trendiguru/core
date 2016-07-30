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

from amazon_signature import get_amazon_signed_url
from time import strftime, gmtime, sleep, time
from requests import get
import xmltodict
from db_utils import log2file, print_error, theArchiveDoorman, progress_bar
from ..Yonti.pymongo_utils import delete_or_and_index
from ..constants import db, redis_conn
from rq import Queue
from datetime import datetime
from amazon_worker import insert_items
from .fanni import plantForests4AllCategories
from .amazon_constants import blacklist, log_name, colors, status_log
from .dl_excel import mongo2xl


forest = Queue('annoy_forest', connection=redis_conn)
today_date = str(datetime.date(datetime.now()))
q = Queue('amazon_worker', connection=redis_conn)

base_parameters = {
    'AWSAccessKeyId': 'AKIAIQJZVKJKJUUC4ETA',
    'AssociateTag': 'fazz0b-20',
    'Version': '2013-08-01',
    'Availability': 'Available',
    'Operation': 'ItemSearch',
    'Service': 'AWSECommerceService',
    'Timestamp': strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
    'ResponseGroup': 'ItemAttributes, OfferSummary,Images'}

# globals
last_time = time()
FashionGender = 'FashionWomen'
error_flag = False
last_price = 3000.00
last_pct =0


def proper_wait(print_flag=False):
    global last_time
    current_time = time()
    time_diff = current_time - last_time
    if time_diff < 1.01:
        sleep(1.01 - time_diff)
        current_time = time()
    if print_flag:
        print ('time diff: %f' % (current_time - last_time))
    last_time = current_time


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


def make_itemsearch_request(pagenum, node_id, min_price, max_price, price_flag=True, print_flag=False, color='',
                            plus_size_flag=False, family_tree='sequoia'):
    global error_flag, last_price

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
    if plus_size_flag:
        parameters['Keywords'] = 'plus-size'
    color_flag = False
    if len(color):
        color_flag=True
        if plus_size_flag:
            parameters['Keywords'] += ',%s' % color
        else:
            parameters['Keywords'] = color

    req = get_amazon_signed_url(parameters, 'GET', False)
    proper_wait()
    res = get(req)
    # last_time = time()
    last_price = min_price
    try:
        if res.status_code != 200:
            msg = 'not 200!'
            error_flag = True
            raise ValueError(msg)

        res_dict = dict(xmltodict.parse(res.text))
        if 'ItemSearchResponse' not in res_dict.keys():
            msg = 'No ItemSearchResponse'
            raise ValueError(msg)

        res_dict = dict(res_dict['ItemSearchResponse']['Items'])
        if 'Errors' in res_dict.keys():
            msg = 'Error'
            error_flag = True
            raise ValueError(msg)

        if 'TotalResults' in res_dict.keys():
            results_count = int(res_dict['TotalResults'])
        else:
            msg = 'no TotalResualts'
            raise ValueError(msg)

        if results_count == 0:
            msg = 'no results for price_range'
            raise ValueError(msg)

        if 'TotalPages' not in res_dict.keys():
            msg = 'no TotalPages in dict keys'
            raise ValueError(msg)

    except Exception as e:
        results_count = 0
        summary = 'Name: %s, PriceRange: %.2f -> %.2f , ResultCount: %s '\
                  % (family_tree, min_price, max_price, e.message)
        if print_flag:
            print_error(e.message)
        if color_flag:
            summary += '(color -> %s)' % color
        log2file(mode='a', log_filename=log_name, message=summary)
        if e.message == 'no TotalResualts':
            results_count=-1
        return [], results_count

    return res_dict, results_count


def process_results(collection_name, pagenum, node_id, min_price, max_price, family_tree, res_dict=None,
                    items_in_page=10, print_flag=False, color='', plus_size_flag=False):
    if pagenum is not 1:
        res_dict, new_item_count = make_itemsearch_request(pagenum, node_id, min_price, max_price,
                                                           print_flag=print_flag, color=color,
                                                           plus_size_flag=plus_size_flag, family_tree=family_tree)
        if new_item_count == -1:
            print ('try again')
            sleep(0.5)
            res_dict, new_item_count = make_itemsearch_request(pagenum, node_id, min_price, max_price,
                                                               print_flag=print_flag, color=color,
                                                               plus_size_flag=plus_size_flag, family_tree=family_tree)
        if new_item_count < 2:
            return -1

    item_list = res_dict['Item']
    q.enqueue(insert_items, args=(collection_name, item_list, items_in_page, print_flag, family_tree,
                                  plus_size_flag), timeout=5400)

    return 0


def iterate_over_pagenums(total_pages, results_count, collection_name, node_id, min_price, max_price, family_tree,
                          res_dict, plus_size_flag , color=''):
    if total_pages == 1:
        num_of_items_in_page = results_count
    else:
        num_of_items_in_page = 10
        process_results(collection_name, 1, node_id, min_price, max_price, family_tree=family_tree,res_dict=res_dict,
                        items_in_page=num_of_items_in_page, color=color, plus_size_flag=plus_size_flag)

    for pagenum in range(2, total_pages + 1):
        if pagenum == total_pages:
            num_of_items_in_page = results_count - 10 * (pagenum - 1)
            if num_of_items_in_page < 2:
                break
        ret = process_results(collection_name, pagenum, node_id, min_price, max_price, family_tree=family_tree,
                              items_in_page=num_of_items_in_page, color=color, plus_size_flag=plus_size_flag)
        if ret < 0:
            return

    summary = 'Name: %s, PriceRange: %.2f -> %.2f , ResultCount: %d ' \
              % (family_tree, min_price, max_price, results_count)
    if len(color):
        summary += '(color -> %s)' % color
    log2file(mode='a', log_filename=log_name, message=summary)


def filter_by_color(collection_name, node_id, price, family_tree, plus_size_flag=False):
    no_results_seq = 0
    for color in colors:
        if no_results_seq > 5:
            break
        res_dict, results_count = make_itemsearch_request(1, node_id, price, price, color=color,
                                                          plus_size_flag=plus_size_flag, family_tree=family_tree)
        if results_count < 1:
            no_results_seq+=1
            continue

        total_pages = int(res_dict['TotalPages'])
        if total_pages < 1:
            no_results_seq += 1
            continue
        iterate_over_pagenums(total_pages, results_count, collection_name, node_id, price, price, family_tree,
                              res_dict, plus_size_flag, color)
        no_results_seq = 0
    return


def get_results(node_id, collection_name='moshe',  price_flag=True, max_price=3000.0, min_price=0.0,
                results_count_only=False, family_tree='moshe', plus_size_flag=False):
    current_last_price = last_price-0.01
    if max_price<current_last_price:
        max_price = current_last_price
    res_dict, results_count = make_itemsearch_request(1, node_id, min_price, max_price, price_flag=price_flag,
                                                      plus_size_flag=plus_size_flag, family_tree=family_tree)
    if results_count_only:
        return results_count

    if results_count < 2:
        return 0

    cache_name = collection_name + '_cache'
    collection_cache = db[cache_name]
    collection_cache.update_one({'node_id': node_id}, {'$set': {'last_max': max_price}})

    total_pages = int(res_dict['TotalPages'])
    color_flag = False
    if results_count > 100:
        # divide more
        diff = truncate_float_to_2_decimal_places(max_price-min_price)
        if diff <= 0.01:
            if results_count > 110:
                color_flag = True  # later we will farther divide by color if it worth it(>150)
            total_pages = 10
        elif diff <= 0.02:
            get_results(node_id, collection_name, min_price=max_price, max_price=max_price, family_tree=family_tree,
                        plus_size_flag=plus_size_flag)
            new_min_price = max_price - 0.01
            get_results(node_id, collection_name, min_price=new_min_price, max_price=new_min_price,
                        family_tree=family_tree, plus_size_flag=plus_size_flag)
            return 0
        else:
            mid_price = (max_price+min_price)/2.0
            mid_price_rounded = truncate_float_to_2_decimal_places(mid_price)
            get_results(node_id, collection_name, min_price=mid_price_rounded, max_price=max_price,
                        family_tree=family_tree, plus_size_flag=plus_size_flag)
            get_results(node_id, collection_name, min_price=min_price, max_price=mid_price_rounded-0.01,
                        family_tree=family_tree, plus_size_flag=plus_size_flag)
            return 0

    iterate_over_pagenums(total_pages, results_count, collection_name, node_id, min_price, max_price, family_tree,
                          res_dict, plus_size_flag=plus_size_flag)

    if color_flag:
        max_rounded = format_price(max_price, True)
        min_rounded = format_price(min_price, True)
        if max_rounded[-2:] != '01':
            filter_by_color(collection_name, node_id, max_price, family_tree=family_tree, plus_size_flag=plus_size_flag)
        if max_rounded != min_rounded:
            filter_by_color(collection_name, node_id, min_price, family_tree=family_tree, plus_size_flag=plus_size_flag)
    return 0


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
        child_name = build_category_tree(child_id, tab, p, False)
        if child_name is None:  # try again
            print ('##################################################################################################')
            child_name = build_category_tree(child_id, tab, p, False)

        leaf['Children']['names'].append((child_id,child_name))

    db.amazon_category_tree.delete_one({'BrowseNodeId': node_id})
    db.amazon_category_tree.insert_one(leaf)
    return name


def delete_if_not_same_id(collection_name, item_id, matches):
    collection = db[collection_name]
    id_to_del = []
    for tmp_item in matches:
        tmp_id = tmp_item['_id']
        if tmp_id == item_id:
            continue
        id_to_del.append(tmp_id)
    if len(id_to_del):
        collection.delete_many({'_id': {'$in': id_to_del}})


def clear_duplicates(name):
    global last_pct
    collection = db[name]
    before = collection.count()
    all_items = collection.find()
    block_size = before/100
    for i, item in enumerate(all_items):
        m, r = divmod(i, block_size)
        if r == 0:
            last_pct = progress_bar(block_size, before, m, last_pct)
        item_id = item['_id']
        keys = item.keys()
        if 'asin' not in keys:
            collection.delete_one({'_id':item_id})
            continue
        asin= item['asin']
        asin_exists = collection.find({'asin':asin})
        if asin_exists.count()>1:
            delete_if_not_same_id(name, item_id, asin_exists)

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
            delete_if_not_same_id(name, item_id, hash_exists)

        img_url = item['images']['XLarge']
        img_url_exists = collection.find({'images.XLarge': img_url})
        if img_url_exists.count() > 1:
            delete_if_not_same_id(name, item_id, img_url_exists)

    print_error('clear duplicates', 'count before : %d\ncount after : %d' % (before, collection.count()))


def download_all(collection_name, gender='Female', del_collection=False, del_cache=False,
                 cat_tree=False, plus_size_flag=False):
    global error_flag, last_price
    collection = db[collection_name]
    cache_name = collection_name+'_cache'
    collection_cache = db[cache_name]

    if cat_tree:
        build_category_tree()

    if del_collection:
        delete_or_and_index(collection_name, ['id', 'img_hash', 'categories', 'download_data.dl_version', 'parent_asin',
                                              'asin', 'images.XLarge', 'features.color'], delete_flag=True)
    if del_cache:
        delete_or_and_index(cache_name, ['node_id'], delete_flag=True)

    # we will need that for querying the category tree
    if gender is 'Female':
        parent_gender = 'Women'
    else:
        parent_gender = 'Men'

    # retrieve all the leaf nodes - assuming the higher branches has too many items
    leafs_cursor = db.amazon_category_tree.find({'Children.count': 0, 'Parents': parent_gender})
    leafs = [x for x in leafs_cursor]  # change the cursor into a list
    iteration = 0
    status_title = 'download started on %s' % today_date
    log2file(mode='w', log_filename=status_log, message=status_title, print_flag=True)

    while len(leafs):
        # the while loop is for retrying failed downloads
        if iteration > 5:
            break
        not_finished = []
        total_leafs = len(leafs)
        # iterate over all leafs and download them one by one
        for x, leaf in enumerate(leafs):
            node_id = leaf['BrowseNodeId']
            cache_exists = collection_cache.find_one({'node_id': node_id})
            last_price = 3000.0
            if cache_exists:
                if cache_exists['last_max'] > 0.00:
                    last_price = cache_exists['last_max']
                    msg = '%d/%d) node id: %s didn\'t finish -> continuing from %.2f' \
                          % (x, total_leafs, node_id, last_price)
                    log2file(mode='a', log_filename=status_log, message=msg, print_flag=True)

                else:
                    msg = '%d/%d) node id: %s already downloaded!' % (x, total_leafs, node_id)
                    log2file(mode='a', log_filename=status_log, message=msg, print_flag=True)
                    continue
            else:
                cache = {'node_id': node_id,
                         'item_count': 0,
                         'new_items': 0,
                         'last_max': last_price}
                collection_cache.insert_one(cache)
            leaf_name = '->'.join(leaf['Parents']) + '->' + leaf['Name']

            try:
                before_count = collection.count()
                get_results(node_id, collection_name, max_price=last_price, results_count_only=False,
                            family_tree=leaf_name, plus_size_flag=plus_size_flag)
                after_count = collection.count()
                new_items_approx = after_count - before_count
                if error_flag:
                    error_flag = False
                    raise ValueError('probably bad request - will be sent for fresh try')
                msg = 'node id: %s download done -> %d new_items downloaded' % (node_id, new_items_approx)
                log2file(mode='a', log_filename=status_log, message=msg, print_flag=True)
                collection_cache.update_one({'node_id': node_id},
                                            {'$set': {'item_count' : after_count,
                                                      'new_items': new_items_approx,
                                                      'last_max': 0.00}})

            except Exception as e:
                msg = 'ERROR', 'node id: %s failed!' % node_id
                log2file(mode='a', log_filename=status_log, message=msg, print_flag=True)
                error_msg = e.message
                log2file(mode='a', log_filename=status_log, message=error_msg, print_flag=True)
                not_finished.append(leaf)

        leafs = not_finished
        do_again = len(leafs)
        if do_again:
            msg = '%d leafs to do again!' % do_again
            log2file(mode='a', log_filename=status_log, message=msg, print_flag=True)
        iteration += 1

    log2file(mode='a', log_filename=status_log, message='DOWNLOAD FINISHED', print_flag=True)
    # clear_duplicates(collection_name)  # add status bar
    print_error('CLEAR DUPLICATES FINISHED')
    theArchiveDoorman(collection_name, instock_limit=7, archive_limit=14)

    # here i duplicate legging to tights/stockings cause we need this category but we dont have items in it
    leggings = collection.find({'categories':'leggings'})
    for leg in leggings:
        leg.pop('_id', None)
        tmp1 = tmp2 = leg
        tmp1['categories'] = 'tights'
        tmp2['categories'] = 'stockings'
        collection.insert_one(tmp1)
        collection.insert_one(tmp2)

    collection_cache.delete_many({})
    message = '%s is Done!' % collection_name
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
    parser.add_argument('-p', '--plus', dest="plus_size", default=False, action='store_true',
                        help='download plus size for amaze-magazine')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    start = time()
    # get user input
    user_input = getUserInput()
    c_c = user_input.country_code
    col_gender = user_input.gender
    plus_size = user_input.plus_size
    delete_all = user_input.delete_all
    delete_cache = user_input.delete_cache
    build_tree = user_input.tree

    # detect & convert gender to our word styling
    gender_upper = col_gender.upper()
    if gender_upper in ['FEMALE', 'WOMEN', 'WOMAN']:
        col_gender = 'Female'
    elif gender_upper in ['MALE', 'MEN', 'MAN']:
        global FashionGender
        FashionGender = 'FashionMen'
        col_gender = 'Male'
    else:
        print("bad input - gender should be only Female or Male ")
        sys.exit(1)

    # verify valid country code
    cc_upper = c_c.upper()
    if cc_upper != 'US':
        print("bad input - for now only working on US")
        sys.exit(1)

    # build collection name and start logging
    if plus_size:
        col_name = 'amaze_%s' % (col_gender)
        title = "@@@ Amaze-Magazine Download @@@"
    else:
        col_name = 'amazon_%s_%s' % (cc_upper, col_gender)
        title = "@@@ Amazon Download @@@"

    title2 = "you choose to update the %s collection" % col_name
    log2file(mode='w', log_filename=log_name, message=title, print_flag=True)
    log2file(mode='a', log_filename=log_name, message=title2, print_flag=True)

    # when a collection format is requested we verify if that was the meaning
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

    # every fresh start its a good idea to build from scratch the category tree
    if build_tree:
        delete_cache = True

    # start the downloading process
    status_full_path = 'collections.' + col_name + '.status'
    db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Working"}})
    download_all(collection_name=col_name, gender=col_gender, del_collection=delete_all,
                 del_cache=delete_cache, cat_tree=build_tree, plus_size_flag=plus_size)

    # after download finished its time to build a new annoy forest
    db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Finishing"}})
    forest_job = forest.enqueue(plantForests4AllCategories, col_name=col_name, timeout=3600)
    while not forest_job.is_finished and not forest_job.is_failed:
        sleep(300)
    if forest_job.is_failed:
        print ('annoy plant forest failed')

    # to add download summery
    end = time()
    duration = end - start
    dl_info = {"date": today_date,
               "dl_duration": duration,
               "store_info": []}

    mongo2xl('amazon_US', dl_info)

    collection = db[col_name]
    notes_full_path = 'collections.' + col_name + '.notes'
    new_items = collection.find({'download_data.first_dl': today_date}).count()
    db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Done",
                                                                  notes_full_path: new_items}})
