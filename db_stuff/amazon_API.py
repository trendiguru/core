import argparse
import sys
from amazon_signature import get_amazon_signed_url
from time import strftime, gmtime, sleep, time
from requests import get
import xmltodict
from db_utils import log2file, print_error, thearchivedoorman, progress_bar, refresh_similar_results, email
from ..constants import db, redis_conn
from rq import Queue
from datetime import datetime
from amazon_worker import insert_items
from .fanni import plantAnnoyForest, reindex_forest
from .amazon_constants import blacklist, colors, status_log, log_dir, plus_sizes, amazon_categories_list
from .dl_excel import mongo2xl
import re

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
last_pct = 0
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


def make_itemsearch_request(pagenum, node_id, min_price, max_price, price_flag=True, print_flag=False, color='',
                            family_tree='sequoia', category=None):
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
    color_flag = False
    if len(color):
        color_flag = True
        parameters['Keywords'] = color

    if category is not None:
        if color_flag:
            parameters['Keywords'] += ',%s' % category
        else:
            parameters['Keywords'] = category

    last_price = min_price
    req = get_amazon_signed_url(parameters, 'GET', False)
    proper_wait()
    res = get(req)
    try:
        if res.status_code != 200:
            err_msg = 'not 200!'
            error_flag = True
            raise ValueError(err_msg)

        res_dict = dict(xmltodict.parse(res.text))
        if 'ItemSearchResponse' not in res_dict.keys():
            err_msg = 'No ItemSearchResponse'
            raise ValueError(err_msg)

        res_dict = dict(res_dict['ItemSearchResponse']['Items'])
        if 'Errors' in res_dict.keys():
            err_msg = 'Error'
            error_flag = True
            raise ValueError(err_msg)

        if 'TotalResults' in res_dict.keys():
            results_count = int(res_dict['TotalResults'])
        else:
            err_msg = 'no TotalResults'
            raise ValueError(err_msg)

        if results_count == 0:
            err_msg = 'no results for price_range'
            raise ValueError(err_msg)

        if 'TotalPages' not in res_dict.keys():
            err_msg = 'no TotalPages in dict keys'
            raise ValueError(err_msg)

    except ValueError as e:
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


def process_results(col_name, pagenum, node_id, min_price, max_price, family_tree, res_dict=None,
                    items_in_page=10, print_flag=False, color='', category=None):
    if pagenum is not 1:
        res_dict, new_item_count = make_itemsearch_request(pagenum, node_id, min_price, max_price,
                                                           print_flag=print_flag, color=color,
                                                           family_tree=family_tree,
                                                           category=category)
        if new_item_count < 1:
            return -1

    item_list = res_dict['Item']
    q.enqueue(insert_items, args=(col_name, item_list, items_in_page, print_flag, family_tree), timeout=5400)

    return 0


def iterate_over_pagenums(total_pages, results_count, col_name, node_id, min_price, max_price, family_tree,
                          res_dict, color='', category=None):
    global last_price
    if total_pages == 1:
        num_of_items_in_page = results_count
    else:
        num_of_items_in_page = 10
    process_results(col_name, 1, node_id, min_price, max_price, family_tree=family_tree, res_dict=res_dict,
                    items_in_page=num_of_items_in_page, color=color, category=category)
    last_price = min_price
    for pagenum in range(2, total_pages + 1):
        if pagenum == total_pages:
            num_of_items_in_page = results_count - 10 * (pagenum - 1)
            if num_of_items_in_page < 2:
                break
        ret = process_results(col_name, pagenum, node_id, min_price, max_price, family_tree=family_tree,
                              items_in_page=num_of_items_in_page, color=color, category=category)
        if ret < 0:
            return

    summary = 'Name: %s, PriceRange: %.2f -> %.2f , ResultCount: %d ' \
              % (family_tree, min_price, max_price, results_count)
    if len(color):
        summary += '(color -> %s)' % color
    log2file(mode='a', log_filename=log_name, message=summary)


def filter_by_color(col_name, node_id, price, family_tree, category=None):
    no_results_seq = 0
    for color in colors:
        if no_results_seq > 5:
            break
        res_dict, results_count = make_itemsearch_request(1, node_id, price, price, color=color,
                                                          family_tree=family_tree,
                                                          category=category)
        if results_count < 1:
            no_results_seq += 1
            continue

        total_pages = int(res_dict['TotalPages'])
        if total_pages < 1:
            no_results_seq += 1
            continue
        iterate_over_pagenums(total_pages, results_count, col_name, node_id, price, price, family_tree,
                              res_dict, color, category=category)
        no_results_seq = 0
    return


def get_results(leaf_id, node_id, col_name='moshe', price_flag=True, max_price=3000.0, min_price=5.0,
                results_count_only=False, family_tree='moshe', category=None):

    current_last_price = last_price-0.01
    if max_price < current_last_price:
        max_price = current_last_price
    res_dict, results_count = make_itemsearch_request(1, node_id, min_price, max_price, price_flag=price_flag,
                                                      family_tree=family_tree,
                                                      category=category)
    if results_count_only:
        return results_count

    if results_count < 2:
        return 0

    db.amazon_category_tree.update_one({'_id': leaf_id}, {'$set': {'LastPrice': max_price}})

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
                        category=category)
            new_min_price = max_price - 0.01
            get_results(leaf_id, node_id, col_name, min_price=new_min_price, max_price=new_min_price,
                        family_tree=family_tree, category=category)
            return 0
        else:
            mid_price = (max_price+min_price)/2.0
            mid_price_rounded = truncate_float_to_2_decimal_places(mid_price)
            get_results(leaf_id, node_id, col_name, min_price=mid_price_rounded, max_price=max_price,
                        family_tree=family_tree, category=category)
            get_results(leaf_id, node_id, col_name, min_price=min_price, max_price=mid_price_rounded - 0.01,
                        family_tree=family_tree, category=category)
            return 0

    iterate_over_pagenums(total_pages, results_count, col_name, node_id, min_price, max_price, family_tree,
                          res_dict, category=category)

    if color_flag:
        max_rounded = format_price(max_price, True)
        min_rounded = format_price(min_price, True)
        if max_rounded[-2:] != '01':
            filter_by_color(col_name, node_id, max_price, family_tree=family_tree,
                            category=category)
        if max_rounded != min_rounded:
            filter_by_color(col_name, node_id, min_price, family_tree=family_tree,
                            category=category)
    return 0


def build_category_tree(parents, root='7141124011', tab=0, delete_collection=True):

    if delete_collection:
        db.amazon_category_tree.delete_many({})

    parameters = base_parameters.copy()
    parameters['Timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    parameters['Operation'] = 'BrowseNodeLookup'
    parameters['ResponseGroup'] = 'BrowseNodeInfo'
    parameters['BrowseNodeId'] = root
    req = get_amazon_signed_url(parameters, 'GET', False)
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
    result_count = get_results(node_id, price_flag=False, results_count_only=True)

    leaf = {'Name': name,
            'BrowseNodeId': node_id,
            'Parents': parents,
            'Children': {'count': len(children),
                         'names': []},
            'TotalResultsExpected': result_count,
            'TotalDownloaded': 0,
            'LastPrice': last_price,
            'Status': 'waiting'}

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
        db.amazon_category_tree.delete_many({'BrowseNodeId': node_id, 'Name': {'$in': ['tights', 'stockings']}})
        for cat_name in ['tights', 'stockings']:
            leaf_tmp = leaf.copy()
            leaf_tmp['Name'] = cat_name
            leaf_tmp['Children']['count'] = 0
            leaf_tmp['Parents'] = p
            db.amazon_category_tree.insert_one(leaf_tmp)
            print('\t\t%s inserted' % cat_name)

    for child in children:
        if 'BrowseNodeId' not in child.keys():
            continue
        child_id = child['BrowseNodeId']
        child_name = build_category_tree(p, child_id, tab, False)
        if child_name is None:
            print_error('try again')
            child_name = build_category_tree(p, child_id, tab, False)

        leaf['Children']['names'].append((child_id, child_name))

    db.amazon_category_tree.delete_one({'BrowseNodeId': node_id, 'Name': name})
    db.amazon_category_tree.insert_one(leaf)
    return name


def clear_duplicates(col_name):
    global last_pct
    collection = db[col_name]
    bef = collection.count()
    all_items = collection.find({}, {'id': 1, 'parent_asin': 1, 'img_hash': 1, 'images.XLarge': 1, 'sizes': 1,
                                     'color': 1, 'p_hash': 1})
    block_size = bef/100
    for i, item in enumerate(all_items):
        m, r = divmod(i, block_size)
        if r == 0:
            last_pct = progress_bar(block_size, bef, m, last_pct)
        item_id = item['_id']
        keys = item.keys()
        if any(x for x in ['id', 'parent_asin', 'img_hash', 'images', 'sizes', 'color', 'p_hash'] if x not in keys):
            # collection.delete_one({'_id':item_id})
            continue
        idx = item['id']
        collection.delete_many({'id': idx, '_id': {'$ne': item_id}})

        parent = item['parent_asin']
        parent_exists = collection.find({'parent_asin': parent, '_id': {'$ne': item_id}},
                                        {'id': 1, 'sizes': 1, 'color': 1})
        if parent_exists:
            id_to_del = []
            current_sizes = item['sizes']
            current_color = item['color']
            for tmp_item in parent_exists:
                tmp_id = tmp_item['_id']
                tmp_color = tmp_item['color']
                if tmp_color != current_color:
                    continue
                tmp_sizes = tmp_item['sizes']
                for size in tmp_sizes:
                    if size not in current_sizes:
                        current_sizes.append(size)
                id_to_del.append(tmp_id)
            if len(id_to_del):
                collection.delete_many({'_id': {'$in': id_to_del}})

        img_hash = item['img_hash']
        collection.delete_many({'img_hash': img_hash, '_id': {'$ne': item_id}})

        p_hash = item['p_hash']
        collection.delete_many({'p_hash': p_hash, '_id': {'$ne': item_id}})

        img_url = item['images']['XLarge']
        collection.delete_many({'images.XLarge': img_url, '_id': {'$ne': item_id}})
    print('')
    print_error('CLEAR DUPLICATES', 'count before : %d\ncount after : %d' % (bef, collection.count()))


def update_drive(col, cc, items_before=None, dl_duration=None):
    items_after = items_new = 0
    for gender in ['Female', 'Male']:
        if col == 'amazon':
            col_name = '%s_%s_%s' %(col, cc, gender)
        else:
            col_name = '%s_%s' % (col, gender)
        collection = db[col_name]
        items_after += collection.count()
        items_new += collection.find({'download_data.first_dl': today_date}).count()
    dl_duration = dl_duration or 'daily-update'
    items_before = items_before or 'daily-update'
    dl_info = {"start_date": today_date,
               "dl_duration": dl_duration,
               "items_before": items_before,
               "items_after": items_after,
               "items_new": items_new}
    mongo2xl(collection_name, dl_info)


def daily_annoy(col_name, categories, all_cats=False):
    collection = db[col_name]
    if not all_cats:
        categories_with_changes = []
        for cat in categories:
            if collection.find({'categories': cat, 'download_data.first_dl': today_date}).count() > 0:
                categories_with_changes.append(cat)
        categories = categories_with_changes

    categories_num = len(categories)
    for c, cat in enumerate(categories):
        forest_job = forest.enqueue(plantAnnoyForest, args=(col_name, cat, 250), timeout=1800)
        while not forest_job.is_finished and not forest_job.is_failed:
            sleep(30)
        if forest_job.is_failed:
            print ('annoy for %s failed' % cat)
        else:
            msg = "%d/%d annoy done!" % (c, categories_num)
            print_error(msg)
    reindex_forest(col_name)


def verify_plus_size(size_list):
    splited_list = []
    for size in size_list:
        size_upper = size.upper()
        split = re.split(r'\(|\)| |-|,', size_upper)
        for s in split:
            splited_list.append(s)
    if 'SMALL' in splited_list:
        return False
    return any(size for size in splited_list if size in plus_sizes)


def update_plus_size_collection(gender, categories, cc='US'):
    amaze_start = time()
    amaze_name = 'amaze_%s' % gender
    amaze = db[amaze_name]
    items_before = 0
    for gender in ['Female', 'Male']:
        col_name = '%s_%s' % ('amaze', gender)
        items_before += db[col_name].count()
    amazon_name = 'amazon_%s_%s' % (cc, gender)
    amazon = db[amazon_name].find()

    for item in amazon:
        idx = item['id']
        # check if already exists in plus collection
        exists = amaze.find({'id': idx}).count()
        if exists:
            continue
        sizes = item['sizes']
        if type(sizes) != list:
            sizes = [sizes]

        its_plus_size = verify_plus_size(sizes)
        if its_plus_size:
            amaze.insert_one(item)

    thearchivedoorman(amaze_name, instock_limit=14, archive_limit=21)
    print_error('ARCHIVE DOORMAN FINISHED')

    daily_annoy(amaze_name, categories)

    refresh_similar_results('amaze')

    amaze_end = time()
    dl_duration = amaze_end - amaze_start
    update_drive(collection_name, cc, items_before, dl_duration)


def daily_amazon_updates(col_name, gender, all_cats=False, cc='US'):
    # redo annoy for categories which has been changed
    daily_annoy(col_name, amazon_categories_list, all_cats)

    # refresh items which has been changed
    refresh_name = 'amazon_%s' % cc
    refresh_similar_results(refresh_name, amazon_categories_list)

    # upload file to drive
    update_drive('amazon', cc)

    # update plus size
    col_upper = col_name.upper()
    print_error('%s DOWNLOAD FINISHED' % col_upper)

    update_plus_size_collection(gender, amazon_categories_list, cc)
    plus = col_upper + ' PLUS SIZE'
    print_error('%s FINISHED' % plus)
    return


def download_all(col_name, gender='Female'):
    global error_flag, last_price, log_name
    collection = db[col_name]

    # we will need that for querying the category tree
    if gender is 'Female':
        parent_gender = 'Women'
    else:
        parent_gender = 'Men'

    # retrieve all the leaf nodes which hadn't been processed yet - assuming the higher branches has too many items
    leafs_cursor = db.amazon_category_tree.find({'Children.count': 0,
                                                 'Parents': parent_gender,
                                                 'Status': {'$ne': 'done'}})

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
                    db.amazon_category_tree.update_one({'_id': leaf_id}, {'$set': {'Status': 'working'}})
                    cache_msg = '%d/%d) node id: %s -> name: %s starting download' \
                                % (x, total_leafs, node_id, name)
                    log2file(mode='a', log_filename=log_name, message=cache_msg, print_flag=True)
                elif last_price_downloaded > 5.00:
                    cache_msg = '%d/%d) node id: %s -> name: %s didn\'t finish -> continuing from %.2f' \
                                % (x, total_leafs, node_id, name, last_price_downloaded)
                    log2file(mode='a', log_filename=log_name, message=cache_msg, print_flag=True)

                else:
                    cache_msg = '%d/%d) node id: %s -> name: %s already downloaded!' % (x, total_leafs, node_id, name)
                    log2file(mode='a', log_filename=log_name, message=cache_msg, print_flag=True)
                    db.amazon_category_tree.update_one({'_id': leaf_id}, {'$set': {'Status': 'done'}})
                    continue

            leaf_name = '->'.join(leaf['Parents']) + '->' + name

            try:
                if name == 'stockings':
                    category_name = 'Stockings'
                elif name == 'tights':
                    category_name = 'Tights'
                else:
                    category_name = None

                before_count = collection.count()
                get_results(leaf_id, node_id, col_name, max_price=last_price_downloaded, results_count_only=False,
                            family_tree=leaf_name, category=category_name)
                after_count = collection.count()
                items_downloaded += after_count - before_count
                db.amazon_category_tree.update_one({'_id': leaf_id}, {'$set': {'TotalDownloaded': items_downloaded}})

                if error_flag:
                    error_flag = False
                    raise NameError('probably bad request - will be sent for fresh try')
                finished_msg = '%d/%d) node id: %s -> name: %s download done -> %d new_items downloaded' \
                               % (x, total_leafs, node_id, name, items_downloaded)
                log2file(mode='a', log_filename=log_name, message=finished_msg, print_flag=True)
                db.amazon_category_tree.update_one({'_id': leaf_id},
                                                   {'$set': {'Status': 'done',
                                                             'LastPrice': 5.00}})
            except NameError as e:
                error_msg1 = 'ERROR! : node id: %s -> name: %s failed!' % (node_id, name)
                log2file(mode='a', log_filename=log_name, message=error_msg1, print_flag=True)
                error_msg2 = e.message
                log2file(mode='a', log_filename=log_name, message=error_msg2, print_flag=True)

        leafs_cursor = db.amazon_category_tree.find({'Children.count': 0,
                                                     'Parents': parent_gender,
                                                     'Status': {'$ne': 'done'}})
        leafs = [x for x in leafs_cursor]
        total_leafs = len(leafs)
        if total_leafs:
            do_again_msg = '%d leafs to do again!' % total_leafs
            log2file(mode='a', log_filename=log_name, message=do_again_msg, print_flag=True)

    log2file(mode='a', log_filename=log_name, message='DOWNLOAD FINISHED', print_flag=True)
    clear_duplicates(col_name)  # add status bar
    thearchivedoorman(col_name, instock_limit=10, archive_limit=30)
    print_error('ARCHIVE DOORMAN FINISHED')

    message = '%s is Done!' % col_name
    log2file(mode='a', log_filename=log_name, message=message, print_flag=True)


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
    download_all(col_name=col_name, gender=gender)

    # after download finished its time to build a new annoy forest
    db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: 'ANNOY'}})
    daily_amazon_updates(col_name, gender, all_cats=True, cc=cc)

    notes_full_path = 'collections.' + col_name + '.notes'
    db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Done",
                                                                  notes_full_path: 'so and so'}})
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
    if cc_upper != 'US':
        print("bad input - for now only working on US")
        sys.exit(1)

    # every fresh start its a good idea to build from scratch the category tree
    if build_tree:
        build_category_tree([])
    elif delete_cache:
        db.amazon_category_tree.update_many({'Children.count': 0},
                                            {'$set': {'LastPrice': 3000.00,
                                                      'Status': 'waiting',
                                                      'TotalDownloaded': 0}})
    else:
        pass

    collection_name = 'amazon_%s' % cc_upper
    if update_drive_only:
        update_drive('Female', cc_upper)
    elif daily:
        daily_amazon_updates('Female', cc_upper)
        daily_amazon_updates('Male', cc_upper)
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