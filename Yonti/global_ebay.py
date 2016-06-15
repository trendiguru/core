"""
playground for testing the ebay API
"""

import requests
import xmltodict
import collections

APPID = 'trendigu-trendigu-PRD-82f871c7c-d6bc287d'
VERSION = '963'
MAIN_CATEGORY = '11450'


def GET(CAT_ID):
    res = requests.get("http://open.api.ebay.com/Shopping?"
                       "callname=GetCategoryInfo"
                       "&appid=" + APPID + \
                       "&version=" + VERSION + \
                       "&CategoryID=" + CAT_ID + \
                       "&IncludeSelector=ChildCategories")
    category_dict = xmltodict.parse(res.text)
    topCategories = []
    for cat in category_dict['GetCategoryInfoResponse']['CategoryArray']['Category']:
        tmp = {'CategoryName': cat['CategoryName'],
               'CategoryId': cat['CategoryID'],
               'LeafCategory': cat['LeafCategory']}
        topCategories.append (tmp )
    return topCategories

def breakItDown(categories_list):
    for x in categories_list:
        return x['CategoryName'], x['CategoryId']

def getTopCategories():
    # find which categories exists in clothing and shoes (11450)
    topCategories = GET(MAIN_CATEGORY)

    top = []
    for cat in topCategories:
        if any([x for x in ['Men','Women'] if x in cat['CategoryName']]):
            cat_list = GET(cat['CategoryId'])
            children = []
            for i,x in enumerate(cat_list):
                if i==0:
                    continue
                tmp= {'name': x['CategoryName'], 'idx':x['CategoryId']}
                children.append(tmp)
            tmp_top = {'name':cat['CategoryName'],'idx':cat['CategoryId'], 'children_count':len(children),'children':children}
            top.append(tmp_top)

    return top

def getBrandHistogram(idx):
    res = requests.get("http://svcs.ebay.com/services/search/FindingService/v1?"
                       "OPERATION-NAME=findItemsAdvanced&SECURITY-APPNAME="+APPID+ \
                       "&RESPONSE-DATA-FORMAT=XML&REST-PAYLOAD&outputSelector=AspectHistogram"
                       "&categoryId=" + idx + \
                       "&paginationInput.entriesPerPage=1&AvailableItemsOnly=true"
                       "&itemFilter(0).name=ListingType&itemFilter(0).value=FixedPrice"
                       "&itemFilter(1).name=HideDuplicateItems&itemFilter(1).value=true"
                       "&itemFilter(2).name=Condition&itemFilter(2).value=New")
    category_dict = xmltodict.parse(res.text)
    brands = []
    if 'aspectHistogramContainer' not in category_dict['findItemsAdvancedResponse'].keys():
        print('break')
        return brands
    for x in category_dict['findItemsAdvancedResponse']['aspectHistogramContainer']['aspect']:
        if type(x) != collections.OrderedDict:
            print('continue')
            continue
        if x['@name']=='Brand':
            for y in x['valueHistogram']:
                # print(y)
                tmp = {'brand':y['@valueName'],
                       'count':y['count']}
                brands.append(tmp)

    return brands


def fill_brand_info(dictlist):
    dictwithbrands = []
    for topCat in dictlist:
        tmp = {'name':topCat['name'],'idx':topCat['idx'], 'children':[]}
        for child in topCat['children']:
            print(child['name'])
            brands_info = getBrandHistogram(child['idx'])
            tmp_child = {'childName':child['name'], 'brands':brands_info}
            tmp['children'].append(tmp_child)
        dictwithbrands.append(tmp)
    return dictwithbrands

def getItemsbyBrand(category_idx,brand,pageNumber):
    res = requests.get("http://svcs.ebay.com/services/search/FindingService/v1?"
                       "OPERATION-NAME=findItemsAdvanced&SERVICE-VERSION=1.12.0"
                       "&SECURITY-APPNAME="+APPID+ \
                       "&RESPONSE-DATA-FORMAT=XML&REST-PAYLOAD"
                       "&categoryId="+ category_idx+ \
                       "&paginationInput.entriesPerPage=100"
                       "&AvailableItemsOnly=true"
                       "&itemFilter(0).name=ListingType&itemFilter(0).value=FixedPrice"
                       "&itemFilter(1).name=HideDuplicateItems&itemFilter(1).value=true"
                       "&itemFilter(2).name=Condition&itemFilter(2).value=New"
                       "&paginationInput.pageNumber="+str(pageNumber)+ \
                       "&aspectFilter.aspectName=Brand&aspectFilter.aspectValueName="+brand)
    allitems = xmltodict.parse(res.text)

def download(allCatsandBrands):



if __name__=='__main__':
    top_categories = getTopCategories()
    brand_info = fill_brand_info(top_categories)
    download(brand_info)
    print ('done!')
else:
    top_categories = getTopCategories()
    brand_info = fill_brand_info(top_categories)


# for x in brand_info:
#     for y in x['children']:
#         print y['brands']
#         raw_input()

"""
1. get all categories - make list of dict - cat + id
2. for each category - get all brand names and count
3. for each brand - retrieve results
4. use all info
** there are items with no brand / homemade

ebay_not_relevant_categories = ['Belts','Scarves', 'Ties',
ebay_not_relevant_categories = ['Backpacks, Bags & Briefcases', 'Belt Buckles', 'Canes & Walking Sticks','Collar Tips',
                                'Gloves & Mittens', 'Handkerchiefs', 'Hats', 'ID & Document Holders',
                                'Key Chains, Rings & Cases', 'Money Clips', 'Organizers & Day Planners',
                                'Sunglasses & Fashion Eyewear', 'Suspenders, Braces', 'Umbrellas',
Wallets
Wristbands
Mixed Items & Lots
break
Other Men's Accessories
Casual Shirts
Dress Shirts
T-Shirts
Athletic Apparel
Blazers & Sport Coats
Coats & Jackets
Jeans
Pants
Shorts
Sleepwear & Robes
Socks
Suits
Sweaters
Sweats & Hoodies
Swimwear
Underwear
Vests
Mixed Items & Lots
Other Men's Clothing
continue
continue
Athletic
Boots
Casual
Dress/Formal
Occupational
Sandals & Flip Flops
Slippers
Mixed Items & Lots
Belt Buckles
Belts
Collar Tips
Fascinators & Headpieces
Gloves & Mittens
Hair Accessories
Handkerchiefs
Hats
ID & Document Holders
Key Chains, Rings & Finders
Organizers & Day Planners
Scarves & Wraps
Shoe Charms
Sunglasses & Fashion Eyewear
Ties
Umbrellas
Wallets
Wristbands
Mixed Items & Lots
continue
continue
Other Women's Accessories
Athletic Apparel
Coats & Jackets
Dresses
Hosiery & Socks
Intimates & Sleep
Jeans
Jumpsuits & Rompers
Leggings
Maternity
Pants
Shorts
Skirts
Suits & Blazers
Sweaters
Sweats & Hoodies
Swimwear
T-Shirts
Tops & Blouses
Vests
Mixed Items & Lots
Other Women's Clothing
Handbags & Purses
Backpacks & Bookbags
Briefcases & Laptop Bags
Diaper Bags
Travel & Shopping Bags
Handbag Accessories
Mixed Items & Lots
Athletic
Boots
Flats & Oxfords
Heels
Occupational
Sandals & Flip Flops
Slippers
Mixed Items & Lots
"""
