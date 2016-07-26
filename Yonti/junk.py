from ..constants import db, fingerprint_weights, histograms_length
from ..NNSearch import distance_Bhattacharyya

collection = db.fanni

def find_n_nearest_neighbors_copy(fingerprint, database, category, number_of_matches, filter_cat=None,
                                  fp_weights= fingerprint_weights, hist_length=histograms_length,
                                  fp_key='fingerprint'):


    distance_function = distance_Bhattacharyya
    # list of tuples with (entry,distance). Initialize with first n distance values
    if filter_cat == None:
        entries = db[database].find({'categories': category},
                                      {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
    else:

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
        top_fp = find_n_nearest_neighbors_copy(fp, database_name, category_id, number_of_results)
        top_sp = find_n_nearest_neighbors_copy(fp, database_name, category_id, number_of_results, field)
        collection.update_one({'_id':idx},{'$set':{'topresults.fp': top_fp,
                                                   'topresults.sp': top_sp}})
        print ('%d/%d' %(x, total))

