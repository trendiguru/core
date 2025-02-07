"""
function fot testing fanni
"""
import platform
import sys
from skimage import io
import logging
import numpy as np
import cv2
if platform.node()=='Bob':
    name = 'Bob'
    import pymongo
    db = pymongo.MongoClient().mydb
else:
    name = 'not Bob'
    from .. import constants
    db = constants.db

import time
import annoy

def euclidean(fp1, fp2):
    """This calculates distance between to arrays using Euclidean distance."""
    if fp1 is not None and fp2 is not None:
        f12 = np.abs(np.array(fp1) - np.array(fp2))
        f12_p = np.power(f12, 2)
        return np.power(np.sum(f12_p), 0.5)
    else:
        print("null fingerprint sent to distance_1_k ")
        logging.warning("null fingerprint sent to distance_1_k ")
        return None


def distance_Bhattacharyya(fp1, fp2):
    """
    This calculates distance between to arrays.
    the first 6 elements are calculated using euclidean distance (When k = .5 this is the same as Euclidean)
    the next elements are the hue, saturation and value histograms - their distances are measured separately
    using the Bhattacharyya distance.
    later all the 4 components are combined with different weights:
    - the first 6 elements get 5%
    - the hue histogram gets 50%
    - the saturation and value histograms share the other 45% equally
    """
    Bhattacharyya = 3
    weights = [0.05, 0.5, 0.225, 0.225]
    hist_length = [180, 255, 255]
    fp_division = [6, 6 + hist_length[0], 6 + hist_length[0] + hist_length[1],
                   6 + hist_length[0] + hist_length[1] + hist_length[2]]

    if fp1 is not None and fp2 is not None:
        first6 = np.abs(np.array(fp1[:fp_division[0]]) - np.array(fp2[:fp_division[0]]))
        first6 = np.power(np.sum(np.power(first6, 1 / 0.5)), 0.5)
        hue_hist1 = np.float32(fp1[fp_division[0]:fp_division[1]])
        hue_hist2 = np.float32(fp2[fp_division[0]:fp_division[1]])
        hue = float(cv2.compareHist(hue_hist1, hue_hist2, Bhattacharyya))
        sat_hist1 = np.float32(fp1[fp_division[1]:fp_division[2]])
        sat_hist2 = np.float32(fp2[fp_division[1]:fp_division[2]])
        sat = float(cv2.compareHist(sat_hist1, sat_hist2, Bhattacharyya))
        val_hist1 = np.float32(fp1[fp_division[2]:])
        val_hist2 = np.float32(fp2[fp_division[2]:])
        val = float(cv2.compareHist(val_hist1, val_hist2, Bhattacharyya))
        score = weights[0] * first6 + weights[1] * hue + weights[2] * sat + weights[3] * val
        return score

    else:
        print("null fingerprint sent to Bhattacharyya ")
        logging.warning("null fingerprint sent to Bhattacharyya ")
        return None


def find_n_nearest_neighbors(target_dict, entries, number_of_matches, distance_function=None):
    distance_function = distance_function or distance_Bhattacharyya
    # list of tuples with (entry,distance). Initialize with first n distance values
    nearest_n = []
    for i, entry in enumerate(entries):
        if i < number_of_matches:
            ent = entry["fingerprint"]
            tar = target_dict["fingerprint"]
            d = distance_function(ent, tar)
            nearest_n.append((entry, d))
        else:
            if i == number_of_matches:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            ent = entry["fingerprint"]
            tar = target_dict["fingerprint"]
            d = distance_function(ent, tar)
            if d < farthest_nearest:
                insert_at = number_of_matches-2
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
    [result[0].pop('fingerprint') for result in nearest_n]
    nearest_n = [result[0] for result in nearest_n]
    return nearest_n


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


def find_occlusion(name):
    collection = db[name]
    items = collection.find({}, {'fingerprint': 1})

    results = []
    b = {'name': 'bhat',
         'range': 25,
         'processTime': 0}
    results.append(b)

    for r in range(1,21):
        dict = {'name': 'euclid ' +str(r),
                'range':r*25,
                'processTime':0,
                'score':0}
        results.append(dict)


    for x,item in enumerate(items):
        print (x)
        if divmod(x-1, 5)[1] == 0:
            for i in range(21):
                print (results[i])
        enteries = db.GangnamStyle_Female.find({'categories':'dress'},{"fingerprint":1})#,"image.XLarge":1})
        b1 = time.time()
        bhat = find_n_nearest_neighbors(item,enteries,25)
        b2 = time.time()
        b2_1 = b2-b1
        results[0]["processTime"] += b2_1

        # print ("bhat length = %d" % len(bhat))
        for num in range(1,21):
            matches= num * 25
            enteries = db.GangnamStyle_Female.find({'categories': 'dress'},{"fingerprint":1})#,"image.XLarge":1})
            e1 = time.time()
            euclid = find_n_nearest_neighbors(item,enteries,number_of_matches=matches, distance_function=euclidean)
            e2 = time.time()
            e2_1 = e2-e1
            results[num]["processTime"]+=e2_1
            # print ("euclid %d length = %d" %(matches, len(euclid)))
            clickList = [e["_id"] for e in euclid]
            score = [m for m in bhat if m["_id"] in clickList ]
            results[num]["score"] += len(score)
            # print len(score)


    for i in range (21):
        print (results[i])


# find_occlusion('fanni')

def build_forest(name, dis_func, tree_count):
    t = annoy.AnnoyIndex(4608, dis_func)
    items = db.testSpacio.find({})
    for x,item in enumerate(items):
        v= item['sp']
        vector = []
        for i in range(6):
            vector+=v[i]
        t.add_item(x,vector)
        db.sp.update_one({'_id':item['_id']},{'$set':{"AnnoyIndex":x}})
    t.build(tree_count)
    t.save(name)

def annoy_search_sp(name,n, dis_func,fingerprint):
    t = annoy.AnnoyIndex(4608, dis_func)
    t.load(name)
    result = t.get_nns_by_vector(fingerprint,n)
    return result

def annoy_search(name,n, dis_func,fingerprint):
    t = annoy.AnnoyIndex(696, dis_func)
    t.load(name)
    result = t.get_nns_by_vector(fingerprint,n)
    return result

def annoy_timings():
    f = open('annoy_results_spacio.txt','a')
    for trees in [250]:
        for method in ['euclidean', 'angular']:
            name = '/home/yonti/test' + str(trees) + method + 'specio.ann'
            t1 = time.time()
            build_forest(name, method, trees )
            t2 = time.time()
            t3 = t2-t1
            print('trees: %d  method: %s  build_time: %d' %(trees, method, t3))
            f.write('\ntrees: %d  method: %s  build_time: %d \n' %(trees, method, t3))
            for dis in [ 'bhat','euclid']:
                items = db.fanni.find({}, {'fingerprint': 1})
                results = []
                b = {'name': dis,
                    'range': 25,
                    'processTime': 0}
                results.append(b)

                for r in range(1, 41):
                    dict = {'name': 'annoy_ ' + str(r),
                            'range': r * 25,
                            'processTime': 0,
                            'score': 0}
                    results.append(dict)

                for x, item in enumerate(items):

                    enteries = db.fanni_testing_db.find({},{"fingerprint": 1,"AnnoyIndex":1})
                    b1 = time.time()
                    if dis is 'bhat':
                        oneByone = find_n_nearest_neighbors(item, enteries, 25)
                    else:
                        oneByone = find_n_nearest_neighbors(item, enteries, 25,distance_function=euclidean)
                    b2 = time.time()
                    b2_1 = b2 - b1
                    results[0]["processTime"] += b2_1

                    for num in range(1, 41):
                        matches = num * 25
                        a1 = time.time()
                        ann = annoy_search(name,matches,method,item['fingerprint'])
                        a2 = time.time()
                        a2_1 = a2 - a1
                        results[num]["processTime"] += a2_1
                        # print ("euclid %d length = %d" %(matches, len(euclid)))
                        score = [m for m in oneByone if m["AnnoyIndex"] in ann]
                        results[num]["score"] += len(score)
                        # print len(score)

                for i in range(41):
                    print (results[i])
                    f.write(str(results[i]))
                    f.write('\n')
                f.write('\n')
    f.close()

def finals():
    '''
    test the conclusions
    linear search top 25/100 vs annoy+linear (100-1000) for both euclidean and angular
    '''
    f = open('finals_100.txt', 'a')

    for trees in [100, 250, 500]:

        for method in ['euclidean', 'angular']:
            name = '/home/yonti/test' + str(trees) + method + '100.ann'
            t1 = time.time()
            build_forest(name, method, trees)
            t2 = time.time()
            t3 = t2 - t1
            print('trees: %d  method: %s  build_time: %d' % (trees, method, t3))
            f.write('\n\ntrees: %d  method: %s  build_time: %d \n' % (trees, method, t3))

        items = db.fanni.find({}, {'fingerprint': 1})

        totalTimeLinear = 0
        totalTimeEuclid = []
        totalTimeAng = []
        totalScoreEuclid = []
        totalScoreAng = []

        for r in range(2, 11):
            totalTimeEuclid.append(0)
            totalTimeAng.append(0)
            totalScoreEuclid.append(0)
            totalScoreAng.append(0)

        for x, item in enumerate(items):
            print(x)
            enteries = db.fanni_testing_db.find({}, {"fingerprint": 1, "AnnoyIndex": 1})
            #25
            b1 = time.time()
            oneByone = find_n_nearest_neighbors(item, enteries, 100)
            b2 = time.time()
            b2_1 = b2 - b1
            # # 100
            # b1 = time.time()
            # oneByone100 = find_n_nearest_neighbors(item, enteries, 100)
            # b2 = time.time()
            # b2_1_100 = b2 - b1
            f.write('\nitem: %d  query_time: %f \n' % (x, b2_1))
            totalTimeLinear +=b2_1

            for r in range(2,11):
                matches = r*100
                for method in ['euclidean', 'angular']:
                    name = '/home/yonti/test' + str(trees) + method + '100.ann'
                    a1 = time.time()
                    ann = annoy_search(name, matches, method, item['fingerprint'])
                    batch = db.fanni_testing_db.find({"AnnoyIndex":{"$in": ann}},{"fingerprint": 1, "AnnoyIndex": 1})
                    a3 = time.time()
                    twoSteps25 = find_n_nearest_neighbors(item, batch, 100)
                    # twoSteps100 = find_n_nearest_neighbors(item, enteries, 100)
                    a2 = time.time()
                    a2_1 =a2-a1
                    top25 = [t["_id"] for t in twoSteps25]
                    score = [m for m in oneByone if m["_id"] in top25]
                    print (str(batch.count())+" : "+str(a3-a1)+" , linear : "+str(a2-a3)+" ,score : "+str(len(score)))

                    # f.write('batchsize: %d  method: %s query_time: %f accuracy: %f %% \n' % (matches, method, a2_1,4*len(score)))
                    if method == 'euclidean':
                        totalScoreEuclid[r-2] += len(score)
                        totalTimeEuclid[r-2] += a2_1
                    else:
                        totalScoreAng[r-2] +=len(score)
                        totalTimeAng[r-2] +=a2_1


        for r in range(2, 11):
            matches = r * 100
            f.write(
            '\nmethod: Euclidean , batchsize: %d   query_time: %f accuracy: %f %% ' % ( matches,  totalTimeEuclid[r-2]/50.0,
                                                                                        totalScoreEuclid[r-2]/50.0))
        f.write('\n')
        for r in range(2, 11):
            matches = r * 100
            f.write(
                '\nmethod: Angular , batchsize: %d   query_time: %f accuracy: %f %% ' % (
                matches, totalTimeAng[r - 2] / 50.0,
                totalScoreAng[r - 2] / 50.0))

    f.close()
