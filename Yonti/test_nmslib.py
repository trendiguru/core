import nmslib_vector
from ...constants import db


def create_index(col_name, catergory):
    space_type = 'cosinesimil'
    space_param = []
    method_name = 'test_'+catergory
    index_name = method_name + '.index'
    index = nmslib_vector.init(
        space_type,
        space_param,
        method_name,
        nmslib_vector.DataType.VECTOR,
        nmslib_vector.DistType.FLOAT)

    all_items_in_category = db[col_name].find({'categories':catergory})

    for idx, item in enumerate(all_items_in_category):
        nmslib_vector.addDataPoint(index, idx, item['fingerprint']['color'])
        item_id = item['_id']
        db[col_name].update_one({'_id':item_id}, {'$set':{'nmslib_index': idx}})
    index_param = ['NN=17', 'initIndexAttempts=3', 'indexThreadQty=4']
    query_time_param = ['initSearchAttempts=3']

    nmslib_vector.createIndex(index, index_param)

    print 'The index is created'

    nmslib_vector.setQueryTimeParams(index, query_time_param)

    print 'Query time parameters are set'

    print "Results for the freshly created index:"

    nmslib_vector.saveIndex(index, index_name)

    print "The index %s is saved" % index_name

    nmslib_vector.freeIndex(index)


def find_top_knn_nmslib(k, query, category):
    space_type = 'cosinesimil'
    space_param = []
    method_name = 'test_' + category
    index_name = method_name + '.index'
    index = nmslib_vector.init(
        space_type,
        space_param,
        method_name,
        nmslib_vector.DataType.VECTOR,
        nmslib_vector.DistType.FLOAT)

    query_time_param = ['initSearchAttempts=3']

    nmslib_vector.loadIndex(index, index_name)

    print "The index %s is loaded" % index_name

    nmslib_vector.setQueryTimeParams(index, query_time_param)

    print 'Query time parameters are set'

    print "Results for the loaded index"

    query_fp = query['fingerprint']['color']
    query_url = query['images']['XLarge']
    print nmslib_vector.knnQuery(index, k, query_fp)
    print query_url

    nmslib_vector.freeIndex(index)

if __name__ == '__main__':
    col = 'ShopStyle_Female'
    q = db[col].find_one({})
    create_index(col, 'dress')
    find_top_knn_nmslib(10, q, 'dress')

