import nmslib_vector
from trendi.constants import db
from time import time


def build_and_save(col_name, category):
    space_type = 'jsdivslow'
    space_param = []
    method_name = 'small_world_rand'
    index_name = col_name + '_' + category + '.index'
    index = nmslib_vector.init(
        space_type,
        space_param,
        method_name,
        nmslib_vector.DataType.VECTOR,
        nmslib_vector.DistType.FLOAT)

    all_items_in_category = db[col_name].find_and_update({'categories': category}, {'$unset': {'nmslib_index':1}})
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
        item_id = item['_id']
        db[col_name].update_one({'_id':item_id}, {'$set':{'nmslib_index': idx}})
    t2 = time()
    print('addDataPoints took %s secs' % str(t2-t1))
    index_param = ['NN=17', 'initIndexAttempts=3', 'indexThreadQty=32']
    query_time_param = ['initSearchAttempts=3']
    nmslib_vector.createIndex(index, index_param)
    t3 = time()
    print('createIndex took %s secs' % str(t3 - t2))

    nmslib_vector.setQueryTimeParams(index, query_time_param)

    nmslib_vector.saveIndex(index, index_name)

    nmslib_vector.freeIndex(index)


