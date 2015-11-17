__author__ = 'yonatan'

import csv
import StringIO
import zipfile
import time
import re
import datetime

import requests
from rq import Queue

from constants import db, flipkart_relevant_categories, flipkart_paperdoll_women, redis_conn
import dailyDBupdate
from fingerprint_core import generate_mask_and_insert

q = Queue('fingerprint_new', connection=redis_conn)

today_date = str(datetime.datetime.date(datetime.datetime.now()))
affiliate_data = '&affid=kyletrend'
flipkart = 'flipkart'
headers = {"FK-Affiliate-Id": "kyletrend", "FK-Affiliate-Token": "74deca1b038141e2996cd3f170445fbb"}
url = "https://affiliate-api.flipkart.net/affiliate/api/kyletrend.json"


def inter2paperdole(list):
    cat = []
    for i in list:
        cat.append(flipkart_paperdoll_women[i])
    return cat[0]  # for now it only return the first match


if __name__ == "__main__":

    if db.download_data.find({"criteria": flipkart}).count() > 0:
        db.download_data.delete_one({"criteria": flipkart})
    db.download_data.insert_one({"criteria": flipkart,
                                 "current_dl": today_date,
                                 "start_time": datetime.datetime.now(),
                                 "items_downloaded": 0,
                                 "new_items": 0,
                                 "errors": 0,
                                 "end_time": "still in process",
                                 "total_dl_time(hours)": "still in process",
                                 "last_request": time.time()})
    db.drop_collection("fp_in_process")
    db.fp_in_process.insert_one({})
    db.fp_in_process.create_index("id")

    r1 = requests.get(url=url, headers=headers)

    r1 = r1.json()

    url2 = r1["apiGroups"]["affiliate"]["rawDownloadListings"]["womens_clothing"]["availableVariants"]["v0.1.0"]["get"]

    r2 = requests.get(url=url2, headers=headers)
    db.download_data.find_one_and_update({"criteria": flipkart},
                                         {'$set': {"start_time": datetime.datetime.now()}})
    total = db.download_data.find({"criteria": flipkart})[0]
    total_time = abs(total["end_time"] - total["start_time"]).total_seconds()
    db.download_data.find_one_and_update({"criteria": flipkart},
                                         {'$set': {"total_dl_time(min)": str(total_time / 60)[:5]}})
    r2zip = zipfile.ZipFile(StringIO.StringIO(r2.content))
    r2zip.extractall()
    csv_file = open(r2zip.infolist()[0].filename, 'rb')
    time.sleep(120)
    DB = csv.reader(csv_file)
    time.sleep(120)

    for x, row in enumerate(DB):
        tmp_prod = {}
        tmp = row[1]
        catgoriz = re.split(" ", tmp)
        inter = [i for i in catgoriz if i in flipkart_relevant_categories]
        if len(inter) > 0:
            tmp_prod["id"] = row[0]
            db.fp_in_process.insert_one({"id": tmp_prod["id"]})
            db.download_data.find_one_and_update({"criteria": flipkart},
                                                 {'$inc': {"items_downloaded": 1}})
            tmp_prod["categories"] = inter2paperdole(inter)
            tmp_prod["clickUrl"] = row[6] + affiliate_data
            imgUrl = re.split(';', row[3])
            tmp_prod["images"] = {'Original': imgUrl[3],
                                  'Best': imgUrl[5],
                                  'IPhone': imgUrl[9],
                                  'IPhoneSmall': imgUrl[8],
                                  'Large': imgUrl[12],
                                  'Medium': imgUrl[11],
                                  'Small': imgUrl[7],
                                  'XLarge': imgUrl[10]}
            tmp_prod["instock"] = row[10]
            tmp_prod["shortDescription"] = row[1]
            tmp_prod["longDescription"] = row[2]
            tmp_prod["price"] = {'price': row[5],
                                 'currency': 'INR'}
            tmp_prod["brand"] = row[8]
            prev = db.flipkart.find_one({'id': tmp_prod["id"]})
            if prev is None:
                q.enqueue(generate_mask_and_insert, doc=tmp_prod, image_url=imgUrl[1],
                          fp_date=today_date, coll=flipkart)
                db.download_data.find_one_and_update({"criteria": flipkart},
                                                     {'$inc': {"new_items": 1}})
            else:
                tmp_prod["fingerprint"] = prev["fingerprint"]
                tmp_prod["download_data"] = prev["download_data"]
                tmp_prod["download_data"]['dl_version'] = today_date
                try:
                    db.flipkart.insert_one(tmp_prod)
                    print "prod inserted successfully"
                    db.fp_in_process.delete_one({"id": tmp_prod["id"]})

                except:
                    db.download_data.find_one_and_update({"criteria": flipkart},
                                                         {'$inc': {"errors": 1}})
                    print "error inserting"
                print ("row #" + str(x))

    print ("sending mail")
    dailyDBupdate.stats_and_mail(flipkart)
    print ("Finished!!!")

"""
flipkart info:
1. headers  = our affiliate info
        FK-Affiliate-Id: kyletrend
        FK-Affiliate-Token: 74deca1b038141e2996cd3f170445fbb
        * the Token can be changed manually through our flipkart account

2.first connection - each time we want to download we need to create the first connection by :
                    get( "https://affiliate-api.flipkart.net/affiliate/api/kyletrend.json")
                    * notice that kyletrend is our id
                   - the returning message is a json containing 2 sub divisions - affiliate link and raw info.
                   - each division contains all the main categories

3. downloading the products info -
        a. affiliate links - each link is a url which is returns ~500 products and a link for the next 500 results
                           - each call takes about 10 sec
                           - the products in the lists alredy contain our affiliate info
        b. raw info        - all the products in the category are downloaded at once as a zip file
                           - the zip file size is about 4GB
                           - inside the zip there is a csv file
                           - the download takes about 10 min
                           - the products don't contain any affiliate info and it needs to be added manually

4. product structure - this is the raw info structure
        a. each row in the csv is a list
            - the first row contains the fields names in their place in the row
        b. the fields are:
                 0.productId
                 1.title - a short description - contains the category - the product class should be taken from here
                 2.description - should be analyzed in the future using nlp
                 3.imageUrlStr - a list of img_urls for different img sizes
                                [125x167, 400x400, 275x275, original, 75x75, 700x700, 125x125,
                                 40x40, 100x100, 200x200, 1100x1360, 180x240, 275x340]
                 4.mrp - it's also the price - needs to understand the difference between the fields
                 5.price - in indian rupee!
                 6.productUrl - without our affiliate info
                 7.categories - usually not specific enough - only 'womens_clothing'
                 8.productBrand
                 9.deliveryTime
                 10.inStock - very important! need to be added to the shopstyle prod info
                 11.codAvailable
                 12.emiAvailable
                 13.offers
                 14.discount
                 15.cashBack
                 16.size
                 17.color
                 18.sizeUnit
                 19.sizeVariants
                 20.colorVariants
                 21.styleCode

5. building collection - loop through the csv rows
                       - split the title and look for relevent categories
                       - if relevant :
                            - check if id already exists
                                -does: get prod fp from current collection
                                -doen't: send product to fp
                            - build prod info
                            - insert prod to new collection
                       - change collections

6. what is the prod info?
                - the generic prod info contains the following fields:
                1.  _id - given by pymongo
                2.  id - productId given by the website
                3.  categories - the relevant category in paperdoll
                4.  clickUrl - productUrl (for flipkart we need to add our affiliate info)
                5.  images -    imgUrlStr divided to 8 categories:
                                Best(900X720),Original,IPhone(360X288),IPhoneSmall(125X100),
                                Large(205X164),Medium(140x112),Small(40X32),XLarge(410X328)
                                * the sizes were taken from shopstyle
                                        - need to convert the flipkart image links to these categories
                                * shopstyle links in "image.sizes.XLarge.url"
                6.  inStock - True/False
                7.  shortDescription - title(flipkart)/name(shopstyle)
                8.  longDescription - description
                9.  price - {'price':
                             'priceLabel': currency(shopstyle)   }
                10. Brand - productBrand(flipkart)/brand.name(shopstyle)
                11. download_data -{'dl_version':
                                    'first_dl':
                                    'fp_version':}
                12. fingerprint


"""
