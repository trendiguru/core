from ..constants import db, fingerprint_weights, histograms_length
from ..NNSearch import distance_Bhattacharyya
import argparse
import sys
import pymongo

db = pymongo.MongoClient().mydb

collection = db.fanni


def find_n_nearest_neighbors_copy(fingerprint, database, category, number_of_matches, new_field=None, query_rule='eq', val=1,
                                  fp_weights= fingerprint_weights, hist_length=histograms_length,
                                  fp_key='fingerprint'):


    distance_function = distance_Bhattacharyya
    # list of tuples with (entry,distance). Initialize with first n distance values
    if new_field == None:
        entries = db[database].find({'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
    else:
        if query_rule == 'eq':
            entries = db[database].find({'categories': category, new_field: val},
                                        {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
        else:
            entries = db[database].find({'categories': category, new_field:{'$lte': val+1, '$gte': val-1}},
                                        {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})

    farthest_nearest = 1
    nearest_n = []
    for i, entry in enumerate(entries):
        if i < number_of_matches:
            ent = entry[fp_key]

            d = distance_function(ent, fingerprint, fp_weights, hist_length)
            nearest_n.append((entry, d))
            farthest_nearest = 1
        else:
            if i == number_of_matches:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            d = distance_function(entry[fp_key], fingerprint, fp_weights, hist_length)
            if d < farthest_nearest:
                insert_at = number_of_matches - 2
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
    [result[0].pop('fingerprint') for result in nearest_n]
    [result[0].pop('_id') for result in nearest_n]
    nearest_n = [result[0] for result in nearest_n]
    return nearest_n


def renew_top_results(database_name, field, rule, category_id='dress', number_of_results=16):

    items = collection.find()
    total = items.count()
    for x, item in enumerate(items):
        idx = item['_id']
        fp = item['fingerprint']
        val = item[field]
        top_fp = find_n_nearest_neighbors_copy(fp, database_name, category_id, number_of_results)
        top_sp = find_n_nearest_neighbors_copy(fp, database_name, category_id, number_of_results, field,rule, val)
        collection.update_one({'_id':idx},{'$set':{'topresults.fp': top_fp,
                                                   'topresults.sp': top_sp}})
        print ('%d/%d' %(x, total))


def getUserInput():
    parser = argparse.ArgumentParser(description='"@@@ Yonatan Comparison Tool @@@')
    parser.add_argument('-f', '--field', dest="field",required=True,
                        help='what field to add to query by besides category')
    parser.add_argument('-r', '--rule', dest="rule", default='eq',
                        help='use eq for equal or range for +_1', required=True)
    parser.add_argument('-d', '--db', dest="db",
                        help='which database to use for the testing', required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # get user input
    user_input = getUserInput()
    field = user_input.field
    rule = user_input.rule
    db_name = user_input.db
    if rule not in ['eq', 'range']:
        print ('bad rule! use only eq or range')
        sys.exit(1)
    renew_top_results(db_name, field, rule)
