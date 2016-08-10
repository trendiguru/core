__author__ = 'jeremy'

from trendi.constants import db

def fp_db():

    url = 'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg'
    result = multilabel_get_final_activations(url)

def copy_db(from_collection,to_collection):
    cursor1 = db.from_collection.find()
    cursor2 = db.to_collection.find()
    print('c1 size:'+cursor1.count())
    print('c2 size:'+cursor2.count())

if __name__ == "__main__":
    copy_db('ShopStyle_Female','fptest_jr_ShopStyle_Female')