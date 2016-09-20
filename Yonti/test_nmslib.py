import nmslib_vector
from ..constants import db
from time import time

def create_index(col_name, category):
    space_type = 'cosinesimil'
    space_param = []
    method_name = 'small_world_rand'
    index_name = method_name + '.index'
    index = nmslib_vector.init(
        space_type,
        space_param,
        method_name,
        nmslib_vector.DataType.VECTOR,
        nmslib_vector.DistType.FLOAT)

    all_items_in_category = db[col_name].find({'categories':category})

    for idx, item in enumerate(all_items_in_category):
        fp = item['fingerprint']
        if type(fp) == list:
            color = fp
        elif type(fp)== dict:
            color = fp['color']
        else:
            print('else')
            continue
        nmslib_vector.addDataPoint(index, idx, color)
        item_id = item['_id']
        db[col_name].update_one({'_id':item_id}, {'$set':{'nmslib_index': idx}})
    print('upto here1')
    index_param = ['NN=17', 'initIndexAttempts=3', 'indexThreadQty=4']
    print('upto here2')
    query_time_param = ['initSearchAttempts=3']

    print('upto here3')

    nmslib_vector.createIndex(index, index_param)

    print 'The index is created'

    nmslib_vector.setQueryTimeParams(index, query_time_param)

    print 'Query time parameters are set'

    print "Results for the freshly created index:"

    nmslib_vector.saveIndex(index, index_name)

    print "The index %s is saved" % index_name

    nmslib_vector.freeIndex(index)

def find_top_knn_nmslib(k, query, category, col_name):
    space_type = 'cosinesimil'
    space_param = []
    method_name = 'small_world_rand'
    index_name = method_name + '.index'
    index = nmslib_vector.init(
        space_type,
        space_param,
        method_name,
        nmslib_vector.DataType.VECTOR,
        nmslib_vector.DistType.FLOAT)
    print('upto here4')
    all_items_in_category = db[col_name].find({'categories':category})

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
        # db[col_name].update_one({'_id': item_id}, {'$set': {'nmslib_index': idx}})
    query_time_param = ['initSearchAttempts=3']
    print('upto here5')

    nmslib_vector.loadIndex(index, index_name)

    print "The index %s is loaded" % index_name

    nmslib_vector.setQueryTimeParams(index, query_time_param)

    print 'Query time parameters are set'

    print "Results for the loaded index"

    query_fp = query['fingerprint']
    if type(query_fp) == list:
        color = query_fp
    elif type(query_fp) == dict:
        color = query_fp['color']
    else:
        print('bad fp')
        return
    query_url = query['images']['XLarge']
    print nmslib_vector.knnQuery(index, k, color)
    print query_url

    nmslib_vector.freeIndex(index)

if __name__ == '__main__':
    a= time()
    col = 'ShopStyle_Female'
    q = db[col].find({'categories': 'dress'})[1000]
    create_index(col, 'dress')
    b = time()
    print ('createtime = %s' %(str(b-a)))
    find_top_knn_nmslib(1000, q, 'dress', col)
    c= time()
    print ('createtime = %s' %(str(c-b)))

