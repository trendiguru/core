import nmslib_vector
from trendi.constants import db
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

    all_items_in_category = db[col_name].find({'categories':category})
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

    return index, nmslib_vector

dress_index, dress_vector = create_index('amazon_US_Female', 'dress')
print "Done initializing!"

# def find_top_knn_nmslib(k, query, category, col_name):
#     space_type = 'cosinesimil'
#     space_param = []
#     method_name = 'small_world_rand'
#     index_name = method_name + '.index'
#     index = nmslib_vector.init(
#         space_type,
#         space_param,
#         method_name,
#         nmslib_vector.DataType.VECTOR,
#         nmslib_vector.DistType.Float)
#     print('upto here4')
#     all_items_in_category = db[col_name].find({'categories':category})
#
#     for idx, item in enumerate(all_items_in_category):
#         # p = item['p_hash']
#         # p_bin = hexa2bin(p)
#         fp = item['fingerprint']
#         if type(fp) == list:
#             color = fp
#         elif type(fp) == dict:
#             color = fp['color']
#         else:
#             print('else')
#             continue
#         nmslib_vector.addDataPoint(index, idx, color)
#         # item_id = item['_id']
#         # db[col_name].update_one({'_id': item_id}, {'$set': {'nmslib_index': idx}})
#     query_time_param = ['initSearchAttempts=5']
#     print('upto here5')
#
#     nmslib_vector.loadIndex(index, index_name)
#
#     print "The index %s is loaded" % index_name
#     t1 = time()
#     nmslib_vector.setQueryTimeParams(index, query_time_param)
#
#     print 'Query time parameters are set'
#
#     print "Results for the loaded index"
#
#     # query_fp = query['fingerprint']
#     # if type(query_fp) == list:
#     #     color = query_fp
#     # elif type(query_fp) == dict:
#     #     color = query_fp['color']
#     # else:
#     #     print('bad fp')
#     #     return
#     p = query['p_hash']
#     p_bin = hexa2bin(p)
#     query_url = query['images']['XLarge']
#     l = nmslib_vector.knnQuery(index, k, p_bin)
#     print query_url
#     t2 = time()
#     print('loop3 = %s' % str(t2 - t1))
#     nmslib_vector.freeIndex(index)
#     return l


def find_to_k(query_fp, k):
    t1 = time()
    if type(query_fp) == list:
        color = query_fp
    elif type(query_fp) == dict:
        color = query_fp['color']
    else:
        print('bad fp')
        return
    # p = query['p_hash']
    # p_bin = hexa2bin(p)
    top_k = dress_vector.knnQuery(dress_index, k, color)

    t2 = time()

    print('loop3 = %s' % str(t2 - t1))
    return top_k
    # print 'Query time parameters are set'
    #
    # print "Results for the freshly created index:"
    #
    #
    # print "The index %s is saved" % index_name
    #
    # nmslib_vector.freeIndex(index)


# if __name__ == '__main__':
#     a= time()
#     col = 'amaze_Female'
#     url = 'http://ecx.images-amazon.com/images/I/41hc8sdCEkL.jpg'
#     q = db[col].find_one({'images.XLarge': url})
#     l = create_index(col, 'blazer', q, 20)
#     # b = time()
#     # print ('createtime = %s' %(str(b-a)))
#     # l= find_top_knn_nmslib(, 'dress', col)
#     c= time()
#     print ('createtime = %s' % (str(c - a)))
#     items = db[col].find({'nmslib_index':{'$in':l}})
#     for item in items:
#         print item['images']['XLarge']



