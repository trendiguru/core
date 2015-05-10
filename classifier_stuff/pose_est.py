import cv2
import pymongo

from matlab_wrapper import matlab_client
import Utils







# matlab = mateng.conn.root.modules
# matlab = mateng.conn.root.matlab
def get_pose_est_bbs(url="http://www.thebudgetbabe.com/uploads/2015/201504/celebsforever21coachella.jpg"):
    mateng = matlab_client.Engine()
    print('got engine')
    # print('7701 is prime?' + str(mateng.isprime(7001)))
    img_arr = Utils.get_cv2_img_array(url, download=True, convert_url_to_local_filename=True)
    if img_arr is None:
        return None
    cv2.imshow('im1', img_arr)
    k = cv2.waitKey(50) & 0xFF

    bbs = mateng.get_pose_boxes_dict(url)
    print('got pose est')
    if bbs is None:
        print('no bps found')
        return None
    # print(bbs)
    # print('human bb ok:'+str(dict['human_bb']))
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0], [255, 255, 255]]
    i = 0
    # for bodypart in bbs:
    # print('bodypart:'+str(bodypart))
    if not 'head' in bbs:
        print('no head found')
        return None
    bodypart = bbs["head"]
    # print('headboxes:' + str(bodypart))
    for bb1 in bodypart:
        bb1 = x1y1x2y2_to_bb(bb1)
        print('rect:' + str(bb1))
        cv2.rectangle(img_arr, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=colors[i], thickness=2)
        cv2.imshow('im1', img_arr)
        k = cv2.waitKey(50) & 0xFF
        i = i + 1

    # cv2.imshow('im1', img_arr)
    # k = cv2.waitKey(0) & 0xFF
    headboxes = bbs['head']
    print('headboxes' + str(headboxes))


def x1y1x2y2_to_bb(x1y1x2y2):
    x1 = x1y1x2y2[0]
    y1 = x1y1x2y2[1]
    x2 = x1y1x2y2[2]
    y2 = x1y1x2y2[3]
    bb = [x1, y1, x2 - x1, y2 - y1]
    return bb


def get_category_from_db(id='v-neck-sweaters'):
    # {"id":"v-neck-sweaters"}
    db = pymongo.MongoClient().mydb
    if db is None:
        print('problem getting DB')
        return None
    num_processes = 100
    # query_doc = {"categories": {"shortName":"V-Necks"}}
    query_doc = {"categories": {"$elemMatch": {"id": id}}}

    # query_doc = {"fp_version": 1}
    fields = {"shortName": 1, "image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1,
              "categories": 1, "id": 1}
    product_cursor = db.products.find(query_doc, fields).batch_size(num_processes)
    TOTAL_PRODUCTS = product_cursor.count()
    print('tot found:' + str(TOTAL_PRODUCTS))
    for i in range(0, TOTAL_PRODUCTS):
        doc = product_cursor[i]
    url = doc['image']['sizes']['XLarge']['url']
    print('neck' + str(i) + ' = ' + str(doc) + ' url:' + str(url))
    get_pose_est_bbs(url=url)
    # query_doc = {"$or": [
    # {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(DB.categories, category_id)}}}},
    # {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(DB.categories, category_id)}}}},

    # {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}
#   ]}
#    else:
#      query_doc = {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}

#   fields = {"image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1, "categories": 1, "id": 1}

# batch_size required because cursor timed out without it. Could use further investigation
#  product_cursor = DB.products.find(query_doc, fields).batch_size(num_processes)
#  TOTAL_PRODUCTS = product_cursor.count()


    # description: classic neckline , round collar, round neck, crew neck, square neck, v-neck, clASsic neckline,round collar,crewneck,crew neck, scoopneck,square neck, bow collar, ribbed round neck,rollneck ,slash neck
    # cats:[{u'shortName': u'V-Necks', u'localizedId': u'v-neck-sweaters', u'id': u'v-neck-sweaters', u'name': u'V-Neck Sweaters'}]
    # cats:[{u'shortName': u'Turtlenecks', u'localizedId': u'turleneck-sweaters', u'id': u'turleneck-sweaters', u'name': u'Turtlenecks'}]
    # cats:[{u'shortName': u'Crewnecks & Scoopnecks', u'localizedId': u'crewneck-sweaters', u'id': u'crewneck-sweaters', u'name': u'Crewnecks & Scoopnecks'}]
    # categories:#            u'name': u'V-Neck Sweaters'}]


    if __name__ == '__main__':
        print('starting')
        # show_all_bbs_in_db()
    # get_pose_est_bbs()
    get_pose_est_bbs()