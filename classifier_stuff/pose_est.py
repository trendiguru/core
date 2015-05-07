import cv2
import pymongo

from matlab_wrapper import matlab_client
import Utils


# matlab = mateng.conn.root.modules
# matlab = mateng.conn.root.matlab
def get_pose_est_bbs():
    mateng = matlab_client.Engine()
    print('got engine')
    print('7701 is prime?' + str(mateng.isprime(7001)))
    url = "http://www.thebudgetbabe.com/uploads/2015/201504/celebsforever21coachella.jpg"
    bbs = mateng.get_pose_boxes_dict(url)
    print('got pose est')
    print(bbs)
    img_arr = Utils.get_cv2_img_array(url, download=True, convert_url_to_local_filename=True)
    if img_arr is None:
        return None
    # print('human bb ok:'+str(dict['human_bb']))
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0], [255, 255, 255]]
    i = 0
    for bodypart in bbs:
        print('bodypart:' + str(bodypart))
        for bb1 in bbs[bodypart]:
            bb1 = x1y1x2y2_to_bb(bb1)
            print('rect:' + str(bb1))
            cv2.rectangle(img_arr, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=colors[i], thickness=2)
            cv2.imshow('im1', img_arr)
            k = cv2.waitKey(50) & 0xFF
        i = i + 1

    cv2.imshow('im1', img_arr)
    k = cv2.waitKey(0) & 0xFF
    headboxes = bbs['head']


def x1y1x2y2_to_bb(x1y1x2y2):
    x1 = x1y1x2y2[0]
    y1 = x1y1x2y2[1]
    x2 = x1y1x2y2[2]
    y2 = x1y1x2y2[3]
    bb = [x1, y1, x2 - x1, y2 - y1]
    return bb


def get_checks_from_db():
    DB = pymongo.MongoClient().mydb


# query_doc = {"$or": [
# {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(DB.categories, category_id)}}}},
#        {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(DB.categories, category_id)}}}},

#            {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}
#   ]}
#    else:
#      query_doc = {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}

#   fields = {"image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1, "categories": 1, "id": 1}

# batch_size required because cursor timed out without it. Could use further investigation
#  product_cursor = DB.products.find(query_doc, fields).batch_size(num_processes)
#  TOTAL_PRODUCTS = product_cursor.count()



if __name__ == '__main__':
    print('starting')
    # show_all_bbs_in_db()
    get_pose_est_bbs()