import cv2

from matlab_wrapper import matlab_client
import Utils
import dbUtils
import background_removal

# matlab = mateng.conn.root.modules
# matlab = mateng.conn.root.matlab
def get_pose_est_bbs(url="http://www.thebudgetbabe.com/uploads/2015/201504/celebsforever21coachella.jpg",
                     description='description', n=0, add_head_rectangle=True, show_visual_output=False):
    mateng = matlab_client.Engine()
    print('got engine')
    # print('7701 is prime?' + str(mateng.isprime(7001)))
    img_arr = Utils.get_cv2_img_array(url, download=True, convert_url_to_local_filename=True)
    if img_arr is None:
        return None
    if show_visual_output:
        cv2.imshow('im1', img_arr)
        k = cv2.waitKey(50) & 0xFF

    # bbs = {'head':[[100,20,130,40],[100,22,130,42]]}
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
        cv2.rectangle(img_arr, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=colors[i], thickness=1)
        if show_visual_output:
            cv2.imshow('im1', img_arr)
            k = cv2.waitKey(50) & 0xFF
        i = i + 1

    # cv2.imshow('im1', img_arr)
    # k = cv2.waitKey(0) & 0xFF
    headboxes = bbs['head']
    boxes_dict = bbs
    box0 = boxes_dict["head"][0]
    box1 = boxes_dict["head"][1]
    box0_w = box0[2] - box0[0]
    box0_h = box0[3] - box0[1]
    box1_w = box1[2] - box1[0]
    box1_h = box1[3] - box1[1]
    a = box0[0]
    b = box1[0]
    avg_x0 = ((a + b) / 2)
    # avg_x0 = np.floor(avg_x0)
    # avg_x0 = np.floor(np.mean([box0[0], box1[0]])).astype(np.uint16)
    #    avg_y0 = np.floor(np.mean([box0[1], box1[1]])).astype(np.uint16)
    avg_y0 = ((box0[1] + box1[1]) / 2)

    #    avg_w = np.floor(np.mean([box0_w, box1_w])).astype(np.uint16)
    avg_w = ((box0_w + box1_w) / 2)
    avg_h = ((box0_h + box1_h) / 2)
    print 'avgw is {0} and avgh is {1}'.format(avg_w, avg_h)
    # avg_h = np.floor(np.mean([box0_h, box1_h])).astype(np.uint16)
    headbox = [int(avg_x0 + (0.05 * avg_w)), int(avg_y0 + (0.05 * avg_h)), int(0.9 * avg_w), int(0.9 * avg_h)]

    #    headbox = pose.pose_est_face(bbs, url)
    print('headboxes' + str(headboxes) + ' headbox' + str(headbox))
    #h = copy.deepcopy(headboxes)
    img_arr = Utils.get_cv2_img_array(url, download=True, convert_url_to_local_filename=True)
    cv2.rectangle(img_arr, (avg_x0, avg_y0), (avg_x0 + avg_w, avg_y0 + avg_h), [0, 0, 100],
                  thickness=1)
    if show_visual_output == True:
        cv2.imshow('im1', img_arr)
        k = cv2.waitKey(200)

    hitbottom = False
    image_h = img_arr.shape[0]
    image_w = img_arr.shape[1]
    w = int(headbox[2])
    h = int(headbox[3])
    top = headbox[1]
    while not hitbottom:
        hitleft_or_right = False
        top = int(top + h)
        if top > image_h:
            hitbottom = True
            break
        left = headbox[0]
        right = headbox[0]
        cv2.rectangle(img_arr, (int(left), int(top)), (int(left + w), int(top + h)), [255, 255, 255],
                      thickness=1)
        while not hitleft_or_right:
            left = int(left - w)
            right = int(right + w)
            print('l {0} r {1} '.format(left, right))
            mid = headbox[0]
            leftbox = [left, top, w, h]
            rightbox = [right, top, w, h]
            if left < 0 or right + w > image_w:
                hitleft_or_right = True
                break
            cv2.rectangle(img_arr, (int(left), int(top)), (int(left + w), int(top + h)), [0, 255, 0],
                          thickness=1)
            cv2.rectangle(img_arr, (right, top), (right + w, top + h), [0, 255, 255],
                          thickness=1)
            if show_visual_output == True:
                cv2.imshow('im1', img_arr)
                k = cv2.waitKey(200)

    cv2.imwrite('images/' + description + "_" + str(n) + ".png", img_arr)
    #return headbox

def x1y1x2y2_to_bb(x1y1x2y2):
    x1 = x1y1x2y2[0]
    y1 = x1y1x2y2[1]
    x2 = x1y1x2y2[2]
    y2 = x1y1x2y2[3]
    bb = [x1, y1, x2 - x1, y2 - y1]
    return bb

if __name__ == '__main__':
    print('starting')
    # show_all_bbs_in_db()
    # get_pose_est_bbs()
    descriptions = ['mermaid', 'A-line', 'shift', 'sheath', 'tent', 'empire', 'strapless', 'halter', 'one-shoulder',
                    'apron',
                    'jumper', 'sun', 'wrap', 'pouf', 'slip', 'qi pao', 'shirt dress', 'maxi', 'ball gown', 'midi',
                    'mini']
    # description = 'mermaid'
 #   get_pose_est_bbs(url="http://www.thebudgetbabe.com/uploads/2015/201504/celebsforever21coachella.jpg",
    # description='description', n=0,add_head_rectangle=True)

    for description in descriptions:
        for i in range(0, 200):
            mdoc = dbUtils.lookfor_next_unbounded_feature_from_db_category(item_number=i, skip_if_marked_to_skip=True,
                                                                           which_to_show='showAll',
                                                                           filter_type='byWordInDescription',
                                                                           category_id=None,
                                                                           word_in_description=description,
                                                                           db=None, )
            if 'doc' in mdoc:
                doc = mdoc['doc']
                print doc

                xlarge_url = doc['image']['sizes']['XLarge']['url']
                print('large img url:' + str(xlarge_url))
                img_arr = Utils.get_cv2_img_array(xlarge_url)
                face1 = background_removal.find_face(img_arr)
                if face1 is not None and len(face1) != 0:
                    print('face1:' + str(face1))
                    bb1 = face1
                else:
                    get_pose_est_bbs(xlarge_url, description, n=i)


#Vnecks round neck natanel
# tight vs nontight dresses yakir
