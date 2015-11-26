__author__ = 'yonatan'

import constants

db = constants.db

collections = ["products", "products_jp"]


def swipe_all():
    for col in collections:
        products = db[col].find()
        for prod in products:
            tmp_prod = []
            id = prod["id"]
            tmp_prod["id"] = id
            cat = prod["categories"][0]["id"]
            if cat not in constants.db_relevant_items:
                db[col].delete_one({"id": id})
                continue
            tmp_prod["categories"] = constants.shopstyle_paperdoll_women[cat]
            tmp_prod["clickUrl"] = prod["clickUrl"]
            tmp_prod["images"] = {'Original': prod['image']['sizes']['Original']['url'],
                                  'Best': prod['image']['sizes']['Best']['url'],
                                  'IPhone': prod['image']['sizes']['IPhone']['url'],
                                  'IPhoneSmall': prod['image']['sizes']['IPhoneSmall']['url'],
                                  'Large': prod['image']['sizes']['Large']['url'],
                                  'Medium': prod['image']['sizes']['Medium']['url'],
                                  'Small': prod['image']['sizes']['Small']['url'],
                                  'XLarge': prod['image']['sizes']['Xlarge']['url']}
            tmp_prod["status"] = {"instock": prod["inStock"],
                                  "hours_out": 0}
            tmp_prod["shortDescription"] = prod["name"]
            tmp_prod["longDescription"] = prod["description"]
            tmp_prod["price"] = {'price': prod["price"],
                                 'currency': prod["currency"]}
            tmp_prod["brand"] = prod["brand"]["name"]
            tmp_prod["download_data"] = prod["download_data"]
            tmp_prod["fingerprint"] = prod["fingerprint"]
            db[col].delete_one({"id": id})
            db[col].insert_one(tmp_prod)


def convert2generic(prod):
    tmp_prod = []
    tmp_prod["id"] = prod["id"]
    cat = prod["categories"][0]["id"]
    tmp_prod["categories"] = constants.shopstyle_paperdoll_women[cat]
    tmp_prod["clickUrl"] = prod["clickUrl"]
    tmp_prod["images"] = {'Original': prod['image']['sizes']['Original']['url'],
                          'Best': prod['image']['sizes']['Best']['url'],
                          'IPhone': prod['image']['sizes']['IPhone']['url'],
                          'IPhoneSmall': prod['image']['sizes']['IPhoneSmall']['url'],
                          'Large': prod['image']['sizes']['Large']['url'],
                          'Medium': prod['image']['sizes']['Medium']['url'],
                          'Small': prod['image']['sizes']['Small']['url'],
                          'XLarge': prod['image']['sizes']['Xlarge']['url']}
    tmp_prod["status"] = {"instock": prod["inStock"],
                          "hours_out": 0}
    tmp_prod["shortDescription"] = prod["name"]
    tmp_prod["longDescription"] = prod["description"]
    tmp_prod["price"] = {'price': prod["price"],
                         'currency': prod["currency"]}
    tmp_prod["brand"] = prod["brand"]["name"]
    tmp_prod["download_data"] = prod["download_data"]

    return tmp_prod
