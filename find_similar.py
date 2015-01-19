__author__ = 'liorsabag'

import json
import sqlsoup
import fingerPrint2 as fp2
from NNSearch import findNNs
import datetime
import pdb
import Utils

def find_with_bb_and_keyword(imageURL, bb, keyword, number_of_results=10):
#    pdb.set_trace()
    db = sqlsoup.SQLSoup("mysql://root:zajCA74Salf3@localhost/trendi2")
    table = db.unique_items_by_image_url
    session = db.session

    query = session.query(table)\
        .filter(table.KEYWORDS.like('%'+keyword+'%'))\
        .filter(table.IMAGEURL.isnot(None))\
        .filter(table.fingerprint.isnot(None))

    db_fingerprint_list = []
    for row in query:
        fp_dict = {}
        fp_dict["id"] = row.iditems
        fp_dict["clothingClass"] = row.class1
        fp_dict["fingerPrintVector"] = json.loads(row.fingerprint)
        fp_dict["imageURL"] = row.IMAGEURL
        fp_dict["buyURL"] = row.BUYURL
        db_fingerprint_list.append(fp_dict)

    #Fingerprint the bounded area
    fgpt = fp2.fp(imageURL, bb)
    targetDict = {"clothingClass":keyword, "fingerPrintVector":fgpt}

#    pdb.set_trace()
    closest_matches = findNNs(targetDict, db_fingerprint_list, number_of_results)
    return fgpt.tolist(), closest_matches

def runFp(imageURL, number_of_results):

    print('in find_similar:runFp:imageURL='+str(imageURL)+' n_results='+str(number_of_results))
    #get db fingerprints
    db = sqlsoup.SQLSoup("mysql://root:zajCA74Salf3@localhost/trendi2")
    #table = db.unique_items_by_fp
    table = db.unique_items_by_image_url
    session = db.session

    #shirts_query = session.query(table)\
    #    .filter(table.KEYWORDS.like('%shirt%'))\
    #    .filter(table.fingerprint.isnot(None))\
    #    .filter(table.class1 == 1)
    
    shirts_query = session.query(table)\
        .filter(table.KEYWORDS.like('%shirt%'))\
        .filter(table.fp_date > datetime.datetime(2014, 10, 3, 8, 0))\
        .filter(table.IMAGEURL.isnot(None))\
        .filter(table.fingerprint.isnot(None))\
        .filter(table.class1 == 1)
    
#        .filter(table.imageURL.isnot(None))\     was added by jeremy 031014


    db_fingerprint_list = []
    for row in shirts_query:
        fp_dict = {}
        fp_dict["id"] = row.iditems
        fp_dict["clothingClass"] = row.class1
        fp_dict["fingerPrintVector"] = json.loads(row.fingerprint)
        fp_dict["imageURL"] = row.IMAGEURL
        fp_dict["buyURL"] = row.BUYURL
        db_fingerprint_list.append(fp_dict)

    c_list, fgpt, BB = fp2.classify_and_fingerprint(imageURL)
#    targetDict = {"clothingClass":c_list[0], "fingerPrintVector":fgpt.tolist()}   why was tolist needed in the first place? isnt fgpt a list??
    targetDict = {"clothingClass":c_list[0], "fingerPrintVector":fgpt}

    # pdb.set_trace()
    closest_matches = findNNs(targetDict, db_fingerprint_list, int(number_of_results))
    return fgpt, closest_matches
    
