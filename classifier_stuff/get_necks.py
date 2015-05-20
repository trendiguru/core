import cv2

import Utils
import dbUtils
import background_removal

if __name__ == '__main__':
    print('starting')
    # show_all_bbs_in_db()
    # get_pose_est_bbs()
    descriptions = ['A-line', 'bodycon', 'tent', 'empire', 'strapless', 'halter', 'one-shoulder', 'apron',
                    'jumper', 'sun', 'wrap', 'pouf', 'slip', 'qi pao', 'shirt dress', 'maxi', 'ball gown', 'midi',
                    'mini']
    # description = 'mermaid'
    # A-line', 'shift', 'sheath',
    # get_pose_est_bbs(url="http://www.thebudgetbabe.com/uploads/2015/201504/celebsforever21coachella.jpg",
    #                   description='description', n=0,add_head_rectangle=True)

    for description in descriptions:
        for i in range(0, 500):
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
                print('xlarge img url:' + str(xlarge_url))
                image = Utils.get_cv2_img_array(xlarge_url)

                mask = background_removal.get_fg_mask(image, bounding_box=None)
                masked = background_removal.get_masked_image(image, mask)

                show_visual_output = False
                if show_visual_output:
                    cv2.imshow('im1', masked)
                    k = cv2.waitKey(500) & 0xFF

                cv2.imwrite('images/bg_removed/' + description + "_" + str(i) + ".png", masked)


# Vnecks round neck natanel
#tight vs nontight dresses yakir
