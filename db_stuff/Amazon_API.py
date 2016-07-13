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
from Amazon_signature import get_amazon_signed_url
from time import strftime,gmtime,sleep
from requests import get
import xmltodict
from ..constants import db


blacklist = ['Jewelry', 'Watches', 'Handbags', 'Accessories', 'Lingerie, Sleep & Lounge', 'Socks & Hosiery',
             'Handbags & Wallets', 'Shops', 'Girls', 'Boys', 'Shoes', 'Underwear', 'Baby', 'Sleep & Lounge',
             'Socks', 'Novelty & More', 'Luggage & Travel Gear', 'Uniforms, Work & Safety', 'Costumes & Accessories',
             'Shoe, Jewelry & Watch Accessories', 'Traditional & Cultural Wear', 'Active Underwear', 'Active Socks',
             'Active Supporters', 'Active Base Layers', 'Sports Bras', 'Athletic Socks', 'Athletic Supporters']

base_parameters = {
    'AWSAccessKeyId': 'AKIAIQJZVKJKJUUC4ETA',
    'AssociateTag': 'fazz0b-20',
    'Version':'2013-08-01',
    'Availability': 'Available',
    'Operation': 'ItemSearch',
    'Service': 'AWSECommerceService',
    'Timestamp': strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
    'ResponseGroup': 'ItemAttributes, OfferSummary,Images'}


def format_price(price_float, period=False):
    """
    input - float
    output - string
    """
    pricex100 = price_float*100
    price_int = int(pricex100)
    price_str = str(price_int)

    # verify 4 character string
    while len(price_str)<4:
        price_str ='0'+price_str

    if period:
        price_str = price_str[:-2]+'.'+price_str[-2:]

    return price_str


def process_results(pagenum, node_id, min_price, max_price, res_dict=None, items_in_page=10):

    if pagenum is not 1:
        parameters = base_parameters.copy()
        parameters['Timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        parameters['SearchIndex'] = 'FashionWomen'
        parameters['BrowseNode'] = node_id
        parameters['ItemPage'] = str(pagenum)
        parameters['MinimumPrice'] = format_price(min_price)
        parameters['MaximumPrice'] = format_price(max_price)

        sleep(1.1)
        res = get(get_amazon_signed_url(parameters, 'GET', False))

        if res.status_code != 200:
            # print ('Bad request!!!')
            return 0

        res_dict = dict(xmltodict.parse(res.text))
        if 'ItemSearchResponse' not in res_dict.keys():
            print ('No ItemSearchResponse')
            return 0

        res_dict = dict(res_dict['ItemSearchResponse']['Items'])

    item_list = res_dict['Item']
    new_item_count = 0
    for x,item in enumerate(item_list):
        if (x+1)>items_in_page:
            break
        # asin = 0
        # parent_asin = 0
        # click_url = 0
        # image = 0
        # price = 0
        # atttibutes=0
        # color = 0
        # sizes = 0
        # short_d = 0
        # long_d = 0
        # features = 0
        try:
            item_keys = item.keys()
            if 'ASIN' not in item_keys:
                continue
            asin = item['ASIN']
            if 'ParentASIN' not in item_keys:
                parent_asin = asin
            else:
                parent_asin = item['ParentASIN']
            click_url = item['DetailPageURL']
            if 'LargeImage' in item_keys:
                image = item['LargeImage']['URL']
            elif 'MediumImage' in item_keys:
                image = item['MediumImage']['URL']
            elif 'SmallImage' in item_keys:
                image = item['SmallImage']['URL']
            else:
                # print('No image')
                continue
            offer = item['OfferSummary']['LowestNewPrice']
            price = {'price': float(offer['Amount'])/100,
                     'currency': offer['CurrencyCode'],
                     'priceLabel': offer['FormattedPrice']}
            atttibutes = item['ItemAttributes']
            attr_keys = atttibutes.keys()
            if 'ClothingSize' in attr_keys:
                clothing_size = atttibutes['ClothingSize']
            elif 'Size' in attr_keys:
                clothing_size = atttibutes['Size']
            else:
                print (attr_keys)
                continue
                # raw_input()
            color = atttibutes['Color']
            sizes = [clothing_size]
            short_d = atttibutes['Title']
            if 'Feature' in attr_keys:
                long_d = ' '.join(atttibutes['Feature'])
            else:
                long_d = ''
            features = {'color': color,
                        'sizes': sizes,
                        'shortDescription': short_d,
                        'longDescription': long_d}

            # print('##################################')
            asin_exists = db.amazon_all.find_one({'asin': asin})
            if asin_exists:
                # print('item exists already!')
                continue

            # print('ooooooooooooooooooooooooooooooooooo')
            parent_asin_exists = db.amazon_all.find_one({'parent_asin': parent_asin, 'features.color': features['color']})
            if parent_asin_exists:
                # print ('parent_asin + color already exists')
                sizes = parent_asin_exists['features']['sizes']
                if clothing_size not in sizes:
                    sizes.append(clothing_size)
                    db.amazon_all.update_one({'_id':parent_asin_exists['_id']}, {'$set':{'features.sizes':sizes}})
                    # print ('added another size to existing item')
                else:
                    pass
                    # print ('parent_asin + color + size already exists ----- %s->%s' % (features['color'], clothing_size))
                continue
            # print('????????????????????????????????????')
            new_item = {'asin': asin,
                        'parent_asin': parent_asin,
                        'clickUrl': click_url,
                        'images':{'XLarge':image},
                        'price': price,
                        'features': features}

            # print 'inserting'
            db.amazon_all.insert_one(new_item)
            # print 'item inserted\n'
            new_item_count +=1

        except:
            print ('---------------problem in the way-------------')
            # print (asin)
            # print(parent_asin)
            # print(click_url)
            # print(image)
            # print(price)
            # print(features)
            # print (atttibutes)
            # print(color)
            # print(sizes)
            # print(short_d)
            # print(long_d)
            # raw_input()
            pass

    return new_item_count


def get_results(node_id, price_flag=True, max_price=10000.0, min_price=0.0, results_count_only=False, name='moshe'):
    parameters = base_parameters.copy()
    parameters['Timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    parameters['SearchIndex'] = 'FashionWomen'
    parameters['BrowseNode'] = node_id
    if not price_flag:
        parameters['ResponseGroup'] = 'SearchBins'
    else:
        parameters['MinimumPrice'] = format_price(min_price)
        parameters['MaximumPrice'] = format_price(max_price)

    sleep(1)
    request_url = get_amazon_signed_url(parameters, 'GET', False)
    res = get(request_url)

    if res.status_code != 200:
        # print ('Bad request!!!')
        return 0

    res_dict = dict(xmltodict.parse(res.text))
    if 'ItemSearchResponse' not in res_dict.keys():
        print ('No ItemSearchResponse')
        return 0

    res_dict = dict(res_dict['ItemSearchResponse']['Items'])
    if 'TotalResults' in res_dict.keys():
        results_count = int(res_dict['TotalResults'])
        if results_count_only:
            return results_count
    else:
        print ('bad query')
        return 0

    if 'Errors' in res_dict.keys() or results_count == 0:
        # print('\nError / no results \n checkout the request: \n %s \n' % request_url)
        return 0

    if results_count > 100:
        mid_price = (max_price+min_price)/2
        if (mid_price-min_price) >= 0.01:
            get_results(node_id, min_price=mid_price, max_price=max_price, name=name)
        if (max_price - mid_price) >= 0.01:
            get_results(node_id, min_price=min_price, max_price=mid_price, name=name)
        return 0

    total_pages = int(res_dict['TotalPages'])+1
    if total_pages==2:
        num_of_items_in_page = results_count
    else:
        num_of_items_in_page=10
    new_items_count = process_results(1,node_id, min_price, max_price, res_dict, items_in_page=num_of_items_in_page)
    for pagenum in range(2,total_pages):
        if pagenum==(total_pages-1):
            num_of_items_in_page = results_count-10*(pagenum-1)
        new_items_count += process_results(pagenum, node_id, min_price, max_price,
                                           items_in_page=num_of_items_in_page)

    print ('Name: %s, PriceRange: %s -> %s , ResultCount: %d (%d)'
           % (name, format_price(min_price, True), format_price(max_price, True), results_count, new_items_count))


def build_category_tree(root='7141124011', tab=0, parents=[], delete_collection=False):

    if delete_collection:
        db.amazon_category_tree.delete_many({})

    parameters = base_parameters.copy()
    parameters['Timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    parameters['Operation'] = 'BrowseNodeLookup'
    parameters['ResponseGroup'] = 'BrowseNodeInfo'
    parameters['BrowseNodeId'] = root
    res = get(get_amazon_signed_url(parameters, 'GET', False))

    if res.status_code != 200 :
        # print ('Bad request!!!')
        return None

    res_dict = dict(xmltodict.parse(res.text))
    if 'BrowseNodeLookupResponse' not in res_dict.keys():
        print ('No BrowseNodeLookupResponse')
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
    result_count = get_results(node_id,price_flag=False, results_count_only=True)

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
        sleep(1.5)
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


def download_all(delete_collection=False):
    collection = db.amazon_all
    # build_category_tree(delete_collection=delete_collection)
    print('starting to download')

    if delete_collection:
        collection.delete_many({})
        indexes = collection.index_information().keys()
        for idx in ['id', 'img_hash', 'categories', 'images.XLarge', 'download_data.dl_version', 'asin', 'parent_asin',
                    'features.color']:
            idx_1 = idx + '_1'
            if idx_1 not in indexes:
                collection.create_index(idx, background=True)

    leafs = db.amazon_category_tree.find({'Children.count': 0})
    for leaf in leafs:
        leaf_name = '->'.join(leaf['Parents']) + '->' + leaf['Name']
        node_id = leaf['BrowseNodeId']
        get_results(node_id, results_count_only=False, name=leaf_name)

download_all(True)