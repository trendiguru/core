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


blacklist = ['Jewelry', 'Watches', 'Handbags', 'Accessories']

base_parameters = {
    'AWSAccessKeyId': 'AKIAIQJZVKJKJUUC4ETA',
    'AssociateTag': 'fazz0b-20',
    'Version':'2013-08-01',
    'Availability': 'Available',
    'Operation': 'ItemSearch',
    'Service': 'AWSECommerceService',
    'Timestamp': strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
    'ResponseGroup': 'ItemAttributes, OfferSummary,Images'}


def get_result_count(node_id):
    parameters = base_parameters.copy()
    parameters['SearchIndex']= 'FashionWomen'
    parameters['ResponseGroup'] = 'SearchBins'
    parameters['BrowseNode'] = node_id
    res = get(get_amazon_signed_url(parameters, 'GET', False))

    if res.status_code != 200:
        # print ('Bad request!!!')
        return 0

    res_dict = dict(xmltodict.parse(res.text))
    if 'ItemSearchResponse' not in res_dict.keys():
        print ('No ItemSearchResponse')
        return 0

    res_dict = dict(res_dict['ItemSearchResponse']['Items'])
    if 'TotalResults' in res_dict.keys():
        return int(res_dict['TotalResults'])
    else:
        print ('bad query')
        return 0


def build_category_tree(root = '7141124011', tab=0, parent='orphan'):
    parameters = base_parameters.copy()
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
    result_count = get_result_count(node_id)
    leaf = {'Name': name,
            'BrowseNodeId': node_id,
            'Parent': parent,
            'Children': {'count': len(children),
                         'names': []},
            'TotalResults': result_count}

    tab_space = '\t' * tab
    print('%sname: %s,  ItemId: %s,  Children: %d , result_count: %d'
          % (tab_space, name, leaf['BrowseNodeId'], leaf['Children']['count'], result_count))

    tab += 1
    for child in children:
        sleep(1.5)
        child_id = child['BrowseNodeId']
        child_name = build_category_tree(child_id, tab, name)
        leaf['Children']['names'].append((child_id,child_name))

    db.amazon_category_tree.insert_one(leaf)
    return name

db.amazon_category_tree.delete_many({})
build_category_tree()