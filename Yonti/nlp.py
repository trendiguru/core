from ..constants import db
import re

def word_counter(collection, delete = False):
    '''
    scrap all the descriptions in shopstyle collection and insert them into a new collection
    if exists -> add to count
    if new - > add to collection

    '''
    if delete:
        db.words.delete_many({})

    if 'word' not in db.words.index_information().keys():
        db.words.create_index('word', background=True)

    items = db[collection].find({},{'shortDescription': 1, 'longDescription': 1})

    for item in items:
        short_d = item['shortDescription']
        long_d  = item['longDescription' ]
        for d in [short_d, long_d]:
            capital_d = d.upper()
            word_list = re.split(r' |-|,|;|:|\.', capital_d)
            for word in word_list:
                exists = db.words.find_one({'word': word})
                if exists:
                    db.words.update_one({'_id': exists['_id']}, {'$inc':{'count':1}})
                else:
                    tmp = {'word':word,
                           'count':1}
                    db.words.insert_one(tmp)