"""
function fot testing fanni
"""
import platform
if platform.node()=='Bob':
    name =  'Bob'
    import pymongo
    db = pymongo.MongoClient().mydb
else:
    name = 'not Bob'
    from .. import constants
    db = constants.db
    fp_weights = constants.fingerprint_weights
    bins = constants.histograms_length
    fp_len = constants.fingerprint_length
import sys
import cv2
from skimage import io


def create_test_collection(name, amount=200):
    img_list = db.images.find({'num_of_people': 1})
    # db.drop_collection(name)
    collection = db[name]
    count=0
    for img in img_list:
        if img['image_urls'][0][0:27] == 'http://www.fashionseoul.com':
            for item in img['people'][1]['items']:
                if item['category']=='dress':
                    dict = {'img_url':img['image_urls'][0],
                            'category': 'dress',
                            'fp': item['fp']}
                    image = io.imread(dict['img_url'])
                    cv2.imshow(str(count), cv2.cvtColor(image, cv2.COLOR_BGR2RGB) )
                    key = cv2.waitKey()
                    if key == 32 :
                        count+=1
                        collection.insert_one(dict)
                        cv2.destroyAllWindows()
                        break
                    cv2.destroyAllWindows()
        if count > amount:
            break


# create_test_collection('fanni',100)

def review_collection(name):
    collection = db[name]
    items = collection.find({},{'img_url':1,'_id':1})
    count=items.count()
    for item in items:
        image = io.imread(item['img_url'])
        img_cvtColor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,d = img_cvtColor.shape
        # print (x,y,z)
        h_resized = 400
        w_resized = w*h_resized/h
        img_resized = cv2.resize(img_cvtColor,(w_resized,h_resized))
        cv2.imshow(str(count), img_resized)
        key = cv2.waitKey()
        if key == 32:
            count -= 1
            collection.delete_one({'_id':item['_id']})
        cv2.destroyAllWindows()

# review_collection('fanni')
if name == 'Bob':
    sys.exit()

from ..NNSearch import find_n_nearest_neighbors,distance_1_k
def find_occlusion(name):
    collection = db[name]
    items = collection.find({}, {'fingerprint': 1, '_id':1})
    for item in items:
        enteries = db.GangnamStyle_Female.find({'categories':'dress'})
        bhat = find_n_nearest_neighbors(item,enteries,100,fp_weights,bins,"fingerprint")
        print bhat
        for num in [100.200,300,400,500]:
            euclid = find_n_nearest_neighbors(item,enteries,number_of_matches=num,
                                                       distance_function=distance_1_k,fp_weights=fp_weights,
                                                       hist_length=bins,fp_key="fingerprint")
            print euclid
            clickList = [e["clickUrl"] for e in euclid]
            score = [m for m in bhat if m["clickUrl"] in clickList ]
            print len(score)/100

        break

find_occlusion('fanni')