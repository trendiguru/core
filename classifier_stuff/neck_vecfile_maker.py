import cv2

from matlab_wrapper import matlab_client
import Utils
import dbUtils
import background_removal


# matlab = mateng.conn.root.modules
# matlab = mateng.conn.root.matlab
def get_pose_est_bbs(url="http://www.thebudgetbabe.com/uploads/2015/201504/celebsforever21coachella.jpg",
                     description='description', n=0, add_head_rectangle=True, show_visual_output=False, bb=None):
    mateng = matlab_client.Engine()
    print('got engine')
    # print('7701 is prime?' + str(mateng.isprime(7001)))
    img_arr = Utils.get_cv2_img_array(url, download=True, convert_url_to_local_filename=True,
                                      download_directory='images')
    if img_arr is None:
        return None
    if show_visual_output:
        cv2.imshow('im1', img_arr)
        k = cv2.waitKey(50) & 0xFF

    # bbs = {'head':[[100,20,130,40],[100,22,130,42]]}
    if bb is None:
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
            if show_visual_output:
                cv2.rectangle(img_arr, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), color=colors[i],
                              thickness=1)
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
        # avg_y0 = np.floor(np.mean([box0[1], box1[1]])).astype(np.uint16)
        avg_y0 = ((box0[1] + box1[1]) / 2)

        # avg_w = np.floor(np.mean([box0_w, box1_w])).astype(np.uint16)
        avg_w = ((box0_w + box1_w) / 2)
        avg_h = ((box0_h + box1_h) / 2)
        print 'avgw is {0} and avgh is {1}'.format(avg_w, avg_h)
        # avg_h = np.floor(np.mean([box0_h, box1_h])).astype(np.uint16)
        headbox = [int(avg_x0 + (0.05 * avg_w)), int(avg_y0 + (0.05 * avg_h)), int(0.9 * avg_w), int(0.9 * avg_h)]

        # headbox = pose.pose_est_face(bbs, url)
        print('headboxes' + str(headboxes) + ' headbox' + str(headbox))
    else:
        headbox = bb

    # h = copy.deepcopy(headboxes)
    img_arr = Utils.get_cv2_img_array(url, download=True, convert_url_to_local_filename=True,
                                      download_directory='images')
    # cv2.rectangle(img_arr, (headbox[0], headbox[1]), (headbox[0] +headbox[2], headbox[1] + headbox[3]), [0, 0, 100],
    # thickness=1)

    if img_arr is None:
        return None
    neck_offset = headbox[3] * 1
    neck_extra_width = int(headbox[2] / 2)
    neck_extra_height = int(headbox[3] / 2)
    image_w = img_arr.shape[1]
    image_h = img_arr.shape[0]
    neck_x = max(0, headbox[0] - neck_extra_width)
    neck_y = max(0, headbox[1] + neck_offset - neck_extra_height)
    neck_w = min(headbox[2] + neck_extra_width * 2, image_w - neck_x - 1)
    neck_h = min(headbox[3] + neck_extra_height * 2, image_h - neck_y - 1)
    neckbox = [neck_x, neck_y, neck_w, neck_h]
    if show_visual_output == True:
        cv2.rectangle(img_arr, (neckbox[0], neckbox[1]), (neckbox[0] + neckbox[2], neckbox[1] + neckbox[3]),
                      [0, 200, 100],
                      thickness=1)
        cv2.imshow('im1', img_arr)
        k = cv2.waitKey(200)

    img2 = img_arr[neckbox[0]:neckbox[0] + neckbox[2], neckbox[0]:neckbox[0] + neckbox[2]]

    new_description = description.replace(' ', '')
    image_name = 'images/jr_necktrain/' + new_description + "_" + str(n) + ".png"
    cv2.imwrite(image_name, img_arr)

    # cv2.imwrite('images/jr_necktrain/' + new_description + "_" + str(n) + ".neck_only.png", img2)
    write_vecfile(neckbox, image_name, datafilename='images/jr_necktrain/bboxes.txt')
    # return headbox


def x1y1x2y2_to_bb(x1y1x2y2):
    x1 = x1y1x2y2[0]
    y1 = x1y1x2y2[1]
    x2 = x1y1x2y2[2]
    y2 = x1y1x2y2[3]
    bb = [x1, y1, x2 - x1, y2 - y1]
    return bb


def write_vecfile(rect, image_name, datafilename='images/jr_necktrain/bboxes.txt'):
    f = open(datafilename, 'a+')
    n_rects = 1
    strn = image_name + '  ' + str(n_rects) + '  '
    strn = strn + str(rect[0]) + ' ' + str(rect[1]) + ' ' + str(rect[2]) + ' ' + str(rect[3]) + '   '
    strn = strn + '\n'
    f.write(strn)
    f.close()  # you can omit in most cases as the destructor will call if
    print(strn)


def find_images(description):
    print('starting to find ' + str(description))
    for i in range(0, 500):
        mdoc = dbUtils.lookfor_next_unbounded_feature_from_db_category(current_item=i, skip_if_marked_to_skip=True,
                                                                       which_to_show='showAll',
                                                                       filter_type='byWordInDescription',
                                                                       category_id=None,
                                                                       word_in_description=description,
                                                                       db=None)

        if mdoc is not None and 'doc' in mdoc:
            doc = mdoc['doc']
            print doc

            xlarge_url = doc['image']['sizes']['XLarge']['url']
            print('large img url:' + str(xlarge_url))
            img_arr = Utils.get_cv2_img_array(xlarge_url)
            if img_arr is None:
                return None
            small_img = background_removal.standard_resize(img_arr, 400)[0]
            show_visual_output = False
            if show_visual_output == True:
                cv2.imshow('im1', img_arr)
                k = cv2.waitKey(200)
                cv2.imshow('smallim1', small_img)
                k = cv2.waitKey(200)
            relevance = background_removal.image_is_relevant(small_img)
            if not relevance:
                print('image is not relevant')
                continue
            print('image is relevant')
            face1 = background_removal.find_face(img_arr)
            if face1 is not None and len(face1) != 0:
                print('face1:' + str(face1))
                bb1 = face1[0]
                get_pose_est_bbs(xlarge_url, description, n=i, bb=bb1)

            else:
                get_pose_est_bbs(xlarge_url, description, n=i)


if __name__ == '__main__':
    print('starting')
    # show_all_bbs_in_db()
    # get_pose_est_bbs()
    descriptions = ['round neck', 'classic neckline', 'round collar', 'round neck', 'crew neck',
                    'square neck', 'v-neck', 'classic neckline',
                    'round collar', 'crewneck', 'crew neck', 'scoopneck', 'square neck', 'bow collar',
                    'ribbed round neck', 'rollneck',
                    'slash neck', 'V-Necks', 'v-neck']

    # 'A-line',
    # 'shift', 'sheath', 'tent', 'empire',
    # description = 'mermaid'
    # get_pose_est_bbs(url="http://www.thebudgetbabe.com/uploads/2015/201504/celebsforever21coachella.jpg",
    # description='description', n=0,add_head_rectangle=True)

    for description in descriptions:
        find_images(description)
# p = Pool(len(descriptions))
# print(p.map(find_images, descriptions))




# Vnecks round neck natanel
# tight vs nontight dresses yakir
