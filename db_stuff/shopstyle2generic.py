__author__ = 'yonatan'

import sys
from .. import constants
from . import shopstyle_constants
db = constants.db

collections = ["products", "products_jp"]


def convert2generic(prod, gender):
    tmp_prod = {}
    id = prod["id"]
    tmp_prod["id"] = id
    tmp = [i["id"] for i in prod["categories"]]
    if gender == 'Female':
        cat = [cat for cat in tmp if cat in shopstyle_constants.shopstyle_relevant_items_Female]
        if "women" in cat:
            cat.remove("women")
        if "womens-clothes" in cat:
            cat.remove("womens-clothes")
        if len(cat) == 0:
            return None
        tmp_prod["categories"] = shopstyle_constants.shopstyle_paperdoll_female[cat[0]]
    elif gender == 'Male':
        cat = [cat for cat in tmp if cat in shopstyle_constants.shopstyle_relevant_items_Male]
        if "women" in cat:
            cat.remove("women")
        if "womens-clothes" in cat:
            cat.remove("womens-clothes")
        if len(cat) == 0:
            return None
        tmp_prod["categories"] = shopstyle_constants.shopstyle_paperdoll_male[cat[0]]
    else:
        print('bad gender')
        sys.exit(1)

    tmp_prod["clickUrl"] = prod["clickUrl"]
    tmp_prod["images"] = {'Original': prod['image']['sizes']['Original']['url'],
                          'Best': prod['image']['sizes']['Best']['url'],
                          'IPhone': prod['image']['sizes']['IPhone']['url'],
                          'IPhoneSmall': prod['image']['sizes']['IPhoneSmall']['url'],
                          'Large': prod['image']['sizes']['Large']['url'],
                          'Medium': prod['image']['sizes']['Medium']['url'],
                          'Small': prod['image']['sizes']['Small']['url'],
                          'XLarge': prod['image']['sizes']['XLarge']['url']}
    tmp_prod["status"] = {"instock": prod["inStock"],
                          "hours_out": 0}
    tmp_prod["shortDescription"] = prod["name"]  # localized name
    tmp_prod["longDescription"] = prod["description"]
    tmp_prod["price"] = {'price': prod["price"],
                         'currency': prod["currency"],
                         "priceLabel": prod["priceLabel"]}

    try:
        tmp_prod["brand"] = prod['brand']['name']
    except:
        tmp_prod["brand"] = prod['brandedName']
    tmp_prod["download_data"] = prod["download_data"]
    return tmp_prod


def swipe_all(col):
    # for col in collections:
    products = db[col].find()
    col1 = 'new_' + col
    for x, prod in enumerate(products):
        print (x)


        # if cat not in constants.db_relevant_items:
        #     db[col].delete_one({"id": id})
        #     continue

        tmp_prod = convert2generic(prod)
        if tmp_prod is None:
            print "bad category"
            continue
        tmp_prod["fingerprint"] = prod["fingerprint"]
        # db[col].delete_one({"id": id})
        db[col1].insert_one(tmp_prod)
    print "finished"


