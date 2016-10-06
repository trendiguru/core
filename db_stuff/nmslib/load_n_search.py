import nmslib_vector
from trendi.constants import db
from time import time
from os.path import isfile
from build_index import build_n_save


def load_index(col_name, category, index_version, reindex=False):
    space_type = 'jsdivslow'
    space_param = []
    method_name = 'small_world_rand'
    index_name = '/usr/bash/indexes/'+col_name + '_' + category +index_version+ '.index'
    file_exists = isfile(index_name)
    if reindex or not file_exists:
        build_n_save(col_name, category, index_version)

    index = nmslib_vector.init(
        space_type,
        space_param,
        method_name,
        nmslib_vector.DataType.VECTOR,
        nmslib_vector.DistType.FLOAT)
    nmslib_index = 'nmslib_index'+index_version
    all_items_in_category = db[col_name].find({'categories': category, nmslib_index: {'$exists': 1}})
    t1 = time()
    for idx, item in enumerate(all_items_in_category):
        fp = item['fingerprint']
        if type(fp) == list:
            color = fp
        elif type(fp) == dict:
            color = fp['color']
        else:
            print('else')
            continue
        nmslib_vector.addDataPoint(index, idx, color)
        # item_id = item['_id']
        # db[col_name].update_one({'_id':item_id}, {'$set': {'nmslib_index': idx}})
    t2 = time()
    print('addDataPoints took %s secs' % str(t2-t1))
    # index_param = ['NN=17', 'initIndexAttempts=3', 'indexThreadQty=32']
    query_time_param = ['initSearchAttempts=3']
    nmslib_vector.loadIndex(index, index_name)
    print "The index %s is loaded" % index_name
    t3 = time()
    print('createIndex took %s secs' % str(t3 - t2))

    nmslib_vector.setQueryTimeParams(index, query_time_param)

    return index, nmslib_vector


def find_to_k(query_fp, k, nmslib_vector, category_index):
    t1 = time()
    if type(query_fp) == list:
        color = query_fp
    elif type(query_fp) == dict:
        color = query_fp['color']
    else:
        print('bad fp')
        return
    top_k = nmslib_vector.knnQuery(category_index, k, color)
    t2 = time()

    print('find took = %s' % str(t2 - t1))
    return top_k
