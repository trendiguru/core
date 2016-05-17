"""
function fot testing fanni
"""
from .. import constants

db = constants.db

def create_test_collection(name, amount):
    img_list = db.images.find({'num_of_people': 1})
    db.drop_collection(name)
    collection = db[name]
    count=0
    for img in img_list:
        if img['image_urls'][0][0:27] == 'http://www.fashionseoul.com':
            for item in img['people'][1]['items']:
                if item['category']=='dress':
                    dict = {'img_url':item['image_urls'][0],
                            'category': 'dress',
                            'fp': item['fp']}
                    count+=1
                    collection.insert_one(dict)
                    break
        if count > amount:
            break
