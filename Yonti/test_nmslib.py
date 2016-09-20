import nmslib_vector
from ..constants import db
from time import time

scale = 16
num_of_bits = 256


def hexa2bin(hexa):
    b = bin(int(hexa, scale))[2:].zfill(num_of_bits)
    p = []
    for i in b:
        p.append(int(i))
    return p


def create_index(col_name, category):
    space_type = 'bit_hamming'
    space_param = []
    method_name = 'small_world_rand'
    index_name = method_name + '.index'
    index = nmslib_vector.init(
        space_type,
        space_param,
        method_name,
        nmslib_vector.DataType.VECTOR,
        nmslib_vector.DistType.INT)

    all_items_in_category = db[col_name].find({'categories':category})
    t1 =time()
    for idx, item in enumerate(all_items_in_category):
        p = item['p_hash']
        p_bin = hexa2bin(p)

        # if type(fp) == list:
        #     color = fp
        # elif type(fp)== dict:
        #     color = fp['color']
        # else:
        #     print('else')
        #     continue
        # print (p_bin)
        nmslib_vector.addDataPoint(index, idx, p_bin)
        item_id = item['_id']
        db[col_name].update_one({'_id':item_id}, {'$set':{'nmslib_index': idx}})
    t2 = time()
    print('loop1 = %s' %str(t2-t1))
    index_param = ['NN=17', 'initIndexAttempts=3', 'indexThreadQty=4']
    print('upto here2')
    query_time_param = ['initSearchAttempts=3']

    print('upto here3')

    nmslib_vector.createIndex(index, index_param)
    t3 = time()
    print('loop2 = %s' % str(t3 - t2))
    print 'The index is created'

    nmslib_vector.setQueryTimeParams(index, query_time_param)

    print 'Query time parameters are set'

    print "Results for the freshly created index:"

    nmslib_vector.saveIndex(index, index_name)

    print "The index %s is saved" % index_name

    nmslib_vector.freeIndex(index)

def find_top_knn_nmslib(k, query, category, col_name):
    space_type = 'bit_hamming'
    space_param = []
    method_name = 'small_world_rand'
    index_name = method_name + '.index'
    index = nmslib_vector.init(
        space_type,
        space_param,
        method_name,
        nmslib_vector.DataType.VECTOR,
        nmslib_vector.DistType.INT)
    print('upto here4')
    all_items_in_category = db[col_name].find({'categories':category})

    for idx, item in enumerate(all_items_in_category):
        p = item['p_hash']
        p_bin = hexa2bin(p)
        # if type(fp) == list:
        #     color = fp
        # elif type(fp) == dict:
        #     color = fp['color']
        # else:
        #     print('else')
        #     continue
        nmslib_vector.addDataPoint(index, idx, p_bin)
        # item_id = item['_id']
        # db[col_name].update_one({'_id': item_id}, {'$set': {'nmslib_index': idx}})
    query_time_param = ['initSearchAttempts=3']
    print('upto here5')

    nmslib_vector.loadIndex(index, index_name)

    print "The index %s is loaded" % index_name
    t1 = time()
    nmslib_vector.setQueryTimeParams(index, query_time_param)

    print 'Query time parameters are set'

    print "Results for the loaded index"

    # query_fp = query['fingerprint']
    # if type(query_fp) == list:
    #     color = query_fp
    # elif type(query_fp) == dict:
    #     color = query_fp['color']
    # else:
    #     print('bad fp')
    #     return
    p = query['p_hash']
    p_bin = hexa2bin(p)
    query_url = query['images']['XLarge']
    print nmslib_vector.knnQuery(index, k, p_bin)
    print query_url
    t2 = time()
    print('loop3 = %s' % str(t2 - t1))
    nmslib_vector.freeIndex(index)

if __name__ == '__main__':
    a= time()
    col = 'amaze_Female'
    q = db[col].find({'categories': 'dress'})[1050]
    create_index(col, 'dress')
    b = time()
    print ('createtime = %s' %(str(b-a)))
    find_top_knn_nmslib(1000, q, 'dress', col)
    c= time()
    print ('createtime = %s' %(str(c-b)))


