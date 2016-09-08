import argparse
import sys
from datetime import datetime
from time import strftime, gmtime, sleep, time

import xmltodict
from requests import get
from rq import Queue

from .signature import get_amazon_signed_url
from .amazon_worker import insert_items
from ...constants import db, redis_conn
from ..general.db_utils import log2file, print_error, email
from .amazon_constants import blacklist, colors, log_dir
from .amazon_post import post_download, daily_amazon_updates, update_drive

today_date = str(datetime.date(datetime.now()))
q = Queue('amazon_worker', connection=redis_conn)
post_q = Queue('amazon_post', connection=redis_conn)
base_parameters = {
    'AWSAccessKeyId': 'AKIAIQJZVKJKJUUC4ETA',
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
log_dir_name = log_dir
log_name = ''


def proper_wait(print_flag=False):
    global last_time
    current_time = time()
    time_diff = current_time - last_time
    if time_diff < 1.001:
        sleep(1.001 - time_diff)
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


def make_itemsearch_request(pagenum, node_id, min_price, max_price, cc, price_flag=True, print_flag=False, color='',
                            family_tree='sequoia', category=None):
    global error_flag, last_price, log_name

    parameters = base_parameters.copy()
    parameters['Timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    if cc == 'US':
        parameters['AssociateTag'] = 'fazz0b-20'
        parameters['SearchIndex'] = FashionGender
    elif cc == 'DE':
        parameters['AssociateTag'] = 'trendiguru-21'
        parameters['SearchIndex'] = 'Apparel'
    else:
        return 0
    parameters['BrowseNode'] = node_id
    if not price_flag:
        parameters['ResponseGroup'] = 'SearchBins'
    else:
        parameters['ItemPage'] = str(pagenum)
        parameters['MinimumPrice'] = format_price(min_price)
        parameters['MaximumPrice'] = format_price(max_price)
    color_flag = False
    if len(color):
        color_flag = True
        parameters['Keywords'] = color

    if category is not None:
        if color_flag:
            parameters['Keywords'] += ',%s' % category
        else:
            parameters['Keywords'] = category

    req = get_amazon_signed_url(parameters, cc, 'GET', False)
    proper_wait()
    res = get(req)
    try:
        if res.status_code != 200:
            err_msg = 'not 200!'
            error_flag = True
            sleep(5)
            raise Warning(err_msg)

        last_price = min_price
        res_dict = dict(xmltodict.parse(res.text))
        if 'ItemSearchResponse' not in res_dict.keys():
            err_msg = 'No ItemSearchResponse'
            raise Warning(err_msg)

        res_dict = dict(res_dict['ItemSearchResponse']['Items'])
        res_keys = res_dict.keys()
        if 'Errors' in res_keys:
            err_msg = 'Error'
            error_flag = True
            raise Warning(err_msg)

        if 'TotalResults' in res_keys:
            results_count = int(res_dict['TotalResults'])
        else:
            err_msg = 'no TotalResults'
            raise Warning(err_msg)

        if results_count == 0:
            err_msg = 'no results for price_range'
            raise Warning(err_msg)

        if 'Item' not in res_keys:
            err_msg = 'no Item keys in results.items'
            raise Warning(err_msg)

        if 'TotalPages' not in res_dict.keys():
            err_msg = 'no TotalPages in dict keys'
            raise Warning(err_msg)

    except Warning as e:
        results_count = 0
        summary = 'Name: %s, PriceRange: %.2f -> %.2f , ResultCount: %s '\
                  % (family_tree, min_price, max_price, e.message)
        if print_flag:
            print_error(e.message)
        if color_flag:
            summary += '(color -> %s)' % color
        log2file(mode='a', log_filename=log_name, message=summary)
        if e.message == 'no TotalResualts':
            results_count = -1
        return [], results_count

    return res_dict, results_count


def process_results(col_name, pagenum, node_id, min_price, max_price, cc, family_tree, res_dict=None,
                    items_in_page=10, print_flag=False, color='', category=None):
    if pagenum is not 1:
        res_dict, new_item_count = make_itemsearch_request(pagenum, node_id, min_price, max_price, cc,
                                                           print_flag=print_flag, color=color,
                                                           family_tree=family_tree,
                                                           category=category)
        if new_item_count < 1:
            return -1

    item_list = res_dict['Item']
    q.enqueue(insert_items, args=(col_name, cc, item_list, items_in_page, print_flag, family_tree), timeout=5400)

    return 0


def iterate_over_pagenums(total_pages, results_count, col_name, node_id, min_price, max_price, cc, family_tree,
                          res_dict, color='', category=None):
    global last_price
    if total_pages == 1:
        num_of_items_in_page = results_count
    else:
        num_of_items_in_page = 10
    process_results(col_name, 1, node_id, min_price, max_price, cc, family_tree=family_tree, res_dict=res_dict,
                    items_in_page=num_of_items_in_page, color=color, category=category)
    last_price = min_price
    for pagenum in range(2, total_pages + 1):
        if pagenum == total_pages:
            num_of_items_in_page = results_count - 10 * (pagenum - 1)
            if num_of_items_in_page < 2:
                break
        ret = process_results(col_name, pagenum, node_id, min_price, max_price, cc, family_tree=family_tree,
                              items_in_page=num_of_items_in_page, color=color, category=category)
        if ret < 0:
            return

    summary = 'Name: %s, PriceRange: %.2f -> %.2f , ResultCount: %d ' \
              % (family_tree, min_price, max_price, results_count)
    if len(color):
        summary += '(color -> %s)' % color
    log2file(mode='a', log_filename=log_name, message=summary)


def filter_by_color(col_name, node_id, price, cc, family_tree, category=None):
    no_results_seq = 0
    for color in colors:
        if no_results_seq > 5:
            break
        res_dict, results_count = make_itemsearch_request(1, node_id, price, price, cc, color=color,
                                                          family_tree=family_tree,
                                                          category=category)
        if results_count < 1:
            no_results_seq += 1
            continue

        total_pages = int(res_dict['TotalPages'])
        if total_pages < 1:
            no_results_seq += 1
            continue
        iterate_over_pagenums(total_pages, results_count, col_name, node_id, price, price, cc, family_tree,
                              res_dict, color, category=category)
        no_results_seq = 0
    return


def get_results(leaf_id, node_id, col_name='moshe', price_flag=True, max_price=3000.0, min_price=5.0,
                results_count_only=False, family_tree='moshe', category=None, cc='US'):

    current_last_price = last_price-0.01
    if max_price < current_last_price:
        max_price = current_last_price
    res_dict, results_count = make_itemsearch_request(1, node_id, min_price, max_price, cc, price_flag=price_flag,
                                                      family_tree=family_tree,
                                                      category=category)
    if results_count_only:
        return results_count

    if results_count < 2:
        return 0

    category_tree_name = 'amazon_%s_category_tree' % cc
    category_tree = db[category_tree_name]
    category_tree.update_one({'_id': leaf_id}, {'$set': {'LastPrice': max_price}})

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
            get_results(leaf_id, node_id, col_name, min_price=max_price, max_price=max_price, family_tree=family_tree,
                        category=category, cc=cc)
            new_min_price = max_price - 0.01
            get_results(leaf_id, node_id, col_name, min_price=new_min_price, max_price=new_min_price,
                        family_tree=family_tree, category=category, cc=cc)
            return 0
        else:
            mid_price = (max_price+min_price)/2.0
            mid_price_rounded = truncate_float_to_2_decimal_places(mid_price)
            get_results(leaf_id, node_id, col_name, min_price=mid_price_rounded, max_price=max_price,
                        family_tree=family_tree, category=category, cc=cc)
            get_results(leaf_id, node_id, col_name, min_price=min_price, max_price=mid_price_rounded - 0.01,
                        family_tree=family_tree, category=category, cc=cc)
            return 0

    iterate_over_pagenums(total_pages, results_count, col_name, node_id, min_price, max_price, cc, family_tree,
                          res_dict, category=category)

    if color_flag:
        max_rounded = format_price(max_price, True)
        min_rounded = format_price(min_price, True)
        if max_rounded[-2:] != '01':
            filter_by_color(col_name, node_id, max_price, cc, family_tree=family_tree,
                            category=category)
        if max_rounded != min_rounded:
            filter_by_color(col_name, node_id, min_price, cc, family_tree=family_tree,
                            category=category)
    return 0


def build_category_tree(parents, cc, root='7141124011', tab=0, delete_collection=True):

    category_tree_name = 'amazon_%s_category_tree' % cc
    category_tree = db[category_tree_name]
    if delete_collection:
        category_tree.delete_many({})

    parameters = base_parameters.copy()
    parameters['Timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    parameters['Operation'] = 'BrowseNodeLookup'
    parameters['ResponseGroup'] = 'BrowseNodeInfo'
    if cc == 'US':
        parameters['AssociateTag'] = 'fazz0b-20'
        parameters['SearchIndex'] = FashionGender
        if not tab:
            root = '7141124011'
    elif cc == 'DE':
        parameters['AssociateTag'] = 'trendiguru-21'
        parameters['SearchIndex'] = 'Apparel'
        if not tab:
            root = '78689031'
    else:
        return 0

    parameters['BrowseNodeId'] = root
    req = get_amazon_signed_url(parameters, cc, 'GET', False)
    proper_wait()
    res = get(req)

    if res.status_code != 200:
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
    result_count = get_results(0, node_id, price_flag=False, results_count_only=True, cc=cc)

    leaf = {'Name': name,
            'BrowseNodeId': node_id,
            'Parents': parents,
            'Children': {'count': len(children),
                         'names': []},
            'TotalResultsExpected': result_count,
            'TotalDownloaded': 0,
            'LastPrice': 3000.00,
            'Status': 'waiting',
            'CurrentRound': 0}

    tab_space = '\t' * tab
    print('%sName: %s,  NodeId: %s,  Children: %d , result_count: %d'
          % (tab_space, name, leaf['BrowseNodeId'], leaf['Children']['count'], result_count))

    tab += 1
    if len(parents) == 0:
        p = [name]
    else:
        p = [x for x in parents]
        p.append(name)

    if node_id == '1040660' or node_id == '1040658':
        category_tree.delete_many({'BrowseNodeId': node_id, 'Name': {'$in': ['tights', 'stockings']}})
        for cat_name in ['tights', 'stockings']:
            leaf_tmp = leaf.copy()
            leaf_tmp['Name'] = cat_name
            leaf_tmp['Children']['count'] = 0
            leaf_tmp['Parents'] = p
            db.amazon_category_tree.insert_one(leaf_tmp)
            print('\t\t%s inserted' % cat_name)

    for child in children:
        try:
            if 'BrowseNodeId' not in child.keys():
                continue
        except EnvironmentError:
            continue
        child_id = child['BrowseNodeId']
        child_name = build_category_tree(p, cc, child_id, tab, False)

        if child_name is None:
            print_error('try again')
            child_name = build_category_tree(p, cc, child_id, tab, False)

        leaf['Children']['names'].append((child_id, child_name))

    category_tree.delete_one({'BrowseNodeId': node_id, 'Name': name})
    category_tree.insert_one(leaf)
    return name


def download_all(col_name, cc, gender):
    global error_flag, last_price, log_name
    collection = db[col_name]

    # we will need that for querying the category tree
    if gender is 'Female':
        if cc == 'DE':
            parent_gender = 'Damen'
        else:
            parent_gender = 'Women'
    else:
        if cc == 'DE':
            parent_gender = 'Herren'
        else:
            parent_gender = 'Men'

    category_tree_name = 'amazon_%s_category_tree' % cc
    category_tree = db[category_tree_name]
    # retrieve all the leaf nodes which hadn't been processed yet - assuming the higher branches has too many items
    leafs_cursor = category_tree.find({'Children.count': 0,
                                       'Parents': parent_gender,
                                       'Status': {'$ne': 'done'},
                                       'CurrentRound': {'$lt': 10}})

    leafs = [x for x in leafs_cursor]  # change the cursor into a list
    status_title = '%s download started on %s' % (col_name, today_date)
    log2file(mode='a', log_filename=log_name, message=status_title, print_flag=True)
    total_leafs = len(leafs)
    while total_leafs:
        # iterate over all leafs and download them one by one
        for x, leaf in enumerate(leafs):
            name = leaf['Name']
            node_id = leaf['BrowseNodeId']
            leaf_id = leaf['_id']
            last_price_downloaded = leaf['LastPrice']
            last_price = last_price_downloaded
            status = leaf['Status']
            items_downloaded = leaf['TotalDownloaded']
            if status != 'done':
                if status == 'waiting':
                    category_tree.update_one({'_id': leaf_id}, {'$set': {'Status': 'working'},
                                                                '$inc': {'CurrentRound': 1}})
                    cache_msg = '%d/%d) node id: %s -> name: %s starting download' \
                                % (x, total_leafs, node_id, name)
                    log2file(mode='a', log_filename=log_name, message=cache_msg, print_flag=True)
                elif last_price_downloaded > 5.00:
                    category_tree.update_one({'_id': leaf_id}, {'$inc': {'CurrentRound': 1}})
                    cache_msg = '%d/%d) node id: %s -> name: %s didn\'t finish -> continuing from %.2f' \
                                % (x, total_leafs, node_id, name, last_price_downloaded)
                    log2file(mode='a', log_filename=log_name, message=cache_msg, print_flag=True)

                else:
                    cache_msg = '%d/%d) node id: %s -> name: %s already downloaded!' % (x, total_leafs, node_id, name)
                    log2file(mode='a', log_filename=log_name, message=cache_msg, print_flag=True)
                    category_tree.update_one({'_id': leaf_id}, {'$set': {'Status': 'done'}})
                    continue

            leaf_name = '->'.join(leaf['Parents']) + '->' + name

            before_count = collection.count({'download_data.dl_version': today_date})
            try:
                if name == 'stockings':
                    category_name = 'Stockings'
                elif name == 'tights':
                    category_name = 'Tights'
                else:
                    category_name = None

                get_results(leaf_id, node_id, col_name, max_price=last_price_downloaded, results_count_only=False,
                            family_tree=leaf_name, category=category_name, cc=cc)

                if error_flag:
                    error_flag = False
                    raise StandardError('probably bad request - will be sent for fresh try')
                after_count = collection.count({'download_data.dl_version': today_date})
                downloaded = after_count - before_count
                finished_msg = '%d/%d) node id: %s -> name: %s download done -> %d new_items downloaded' \
                               % (x, total_leafs, node_id, name, downloaded)
                log2file(mode='a', log_filename=log_name, message=finished_msg, print_flag=True)
                category_tree.update_one({'_id': leaf_id},
                                                   {'$set': {'Status': 'done',
                                                             'LastPrice': 5.00}})
            except StandardError as e:
                error_msg1 = 'ERROR! : node id: %s -> name: %s failed!' % (node_id, name)
                log2file(mode='a', log_filename=log_name, message=error_msg1, print_flag=True)
                error_msg2 = e.message
                log2file(mode='a', log_filename=log_name, message=error_msg2, print_flag=True)

            after_count = collection.count({'download_data.dl_version': today_date})
            items_downloaded += after_count - before_count
            category_tree.update_one({'_id': leaf_id}, {'$set': {'TotalDownloaded': items_downloaded}})

        leafs_cursor = category_tree.find({'Children.count': 0,
                                           'Parents': parent_gender,
                                           'Status': {'$ne': 'done'}})
        leafs = [x for x in leafs_cursor]
        total_leafs = len(leafs)
        if total_leafs:
            do_again_msg = '%d leafs to do again!' % total_leafs
            log2file(mode='a', log_filename=log_name, message=do_again_msg, print_flag=True)

    log2file(mode='a', log_filename=log_name, message='DOWNLOAD FINISHED', print_flag=True)


def download_by_gender(gender, cc):
    global log_dir_name, log_name
    # build collection name and start logging
    col_name = 'amazon_%s_%s' % (cc, gender)
    title = "@@@ Amazon %s %s Download @@@" % (cc, gender)

    # TODO: add top level log
    log_name = log_dir_name + col_name + '.log'
    title2 = "you choose to update the %s collection" % col_name
    log2file(mode='w', log_filename=log_name, message=title, print_flag=True)
    log2file(mode='a', log_filename=log_name, message=title2, print_flag=True)

    # start the downloading process
    status_full_path = 'collections.' + col_name + '.status'
    db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Working"}})
    download_all(col_name=col_name, cc=cc, gender=gender)
    db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: 'POST'}})
    post_q.enqueue(post_download, args=(col_name, gender, cc, log_name), timeout=10000)

    return


def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ Amazon Download @@@')
    parser.add_argument('-c', '--code', default="US", dest="country_code",
                        help='country code - currently doing only US')
    parser.add_argument('-g', '--gender', dest="gender",
                        help='specify which gender to download (Female, Male, def=Both)', default='Both')
    parser.add_argument('-f', '--fresh', dest="delete_cache", default=False, action='store_true',
                        help='delete all cache and start a fresh download')
    parser.add_argument('-t', '--tree', dest="tree", default=False, action='store_true',
                        help='build category tree from scratch')
    parser.add_argument('-u', '--updatedrive', dest="update_only", default=False, action='store_true',
                        help='only update the drive')
    parser.add_argument('-d', '--daily', dest="daily_update", default=False, action='store_true',
                        help='daily update - run annoy, reindex, update plus size and upload to drive')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # get user input
    user_input = get_user_input()
    c_c = user_input.country_code
    col_gender = user_input.gender
    delete_cache = user_input.delete_cache
    build_tree = user_input.tree
    update_drive_only = user_input.update_only
    daily = user_input.daily_update
    # verify valid country code
    cc_upper = c_c.upper()
    if cc_upper != 'US' and cc_upper != 'DE':
        print("bad input - for now only working on US and DE")
        sys.exit(1)

    collection_name = 'amazon_%s' % cc_upper
    tree_name = 'amazon_%s_category_tree' % cc_upper
    tree_collection = db[tree_name]
    # every fresh start its a good idea to build from scratch the category tree
    if build_tree:
        build_category_tree([], cc_upper)
        tree_collection.delete_many({'Name': 'Clothing'})

    elif delete_cache:
        tree_collection.update_many({'Children.count': 0},
                                    {'$set': {'LastPrice': 3000.00,
                                              'Status': 'waiting',
                                              'TotalDownloaded': 0}})
    else:
        pass

    if update_drive_only:
        update_drive('amazon', cc_upper)
    elif daily:
        for gen in ['Female', 'Male']:
            col = 'amazon_%s_%s' % (cc_upper, gen)
            daily_amazon_updates(col, gen, cc=cc_upper)
    else:
        # detect & convert gender to our word styling
        gender_upper = col_gender.upper()
        if gender_upper == 'BOTH':
            download_by_gender('Female', cc_upper)
            FashionGender = 'FashionMen'
            download_by_gender('Male', cc_upper)

        elif gender_upper in ['FEMALE', 'WOMEN', 'WOMAN']:
            download_by_gender('Female', cc_upper)

        elif gender_upper in ['MALE', 'MEN', 'MAN']:
            FashionGender = 'FashionMen'
            download_by_gender('Male', cc_upper)

        else:
            print("bad input - gender should be only Female, Male or Both ")

        email(collection_name)

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