__author__ = 'jeremy'
from trendi.yonatan import yonatan_constants
from trendi import Utils
import cv2
import numpy as np

def deepfashion_to_db(attribute_file='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_img.txt',
                        bbox_file='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_bbox.txt',
 #                       bucket='https://tg-training.storage.googleapis.com/deep_fashion/category_and_attribute_prediction/',
                        bucket = 'gs://tg-training/deep_fashion/',
                        use_visual_output=True):

    '''  takes deepfashion list of bbs and attrs, and images in bucket.
    puts bb, attr, and link to file on bucket into db
    apparently mogno / pymongo creates the _id field so i dont have to
    :return:

    '''
    with open(attribute_file,'r') as fp:
        lines = fp.readlines()
        for line in lines[2:]:  #1st line is # of files, 2nd line describes fields 
            attribute_list = []
            #print line
            path = line.split()[0]
            vals = [int(i)+1 for i in line.split()[1:]]  #the vals are -1, 1 so add 1 to get 0, 2
            non_zero_idx = np.nonzero(vals)
            #print non_zero_idx
            for i in range(len(non_zero_idx[0])):
                #print yonatan_constants.attribute_type_dict[str(non_zero_idx[0][i])]
                attribute_list.append(yonatan_constants.attribute_type_dict[str(non_zero_idx[0][i])])
            print('attributes:'+str(attribute_list))
            url = bucket+path
            print('url:'+str(url))
            if use_visual_output:
                img_arr = Utils.get_cv2_img_array(url)
                cv2.imshow(str(attribute_list),img_arr)
                cv2.waitKey()


        return

    imgfiles = [f for f in os.listdir(dir_of_images) if os.path.isfile(os.path.join(dir_of_images,f)) and f[-4:]=='.jpg' or f[-5:]=='.jpeg' ]
    for imgfile in imgfiles:
        corresponding_bbfile=imgfile.split('photo_')[1]
        corresponding_bbfile=corresponding_bbfile.split('.jpg')[0]
        corresponding_bbfile = corresponding_bbfile + '.txt'
        full_filename = os.path.join(dir_of_bbfiles,corresponding_bbfile)
        print('img {} bbfile {} full {}'.format(imgfile,corresponding_bbfile,full_filename))
        full_imgname = os.path.join(dir_of_images,imgfile)
        img_arr = cv2.imread(full_imgname)
        h,w = img_arr.shape[0:2]
#            master_bb.write(full_filename+' '+str(w)+' ' +str(h)+'\n')
        info_dict = {}
        info_dict['url'] = full_imgname
        info_dict['image_width'] = w
        info_dict['image_height'] = h
        items = []
        with open(full_filename,'r+') as fp:
            n_boxes = 0
            for line in fp:
                item_dict = {}
                cv2.imwrite
                n_boxes += 1
             #   line = str(category_number)+' '+str(  dark_bb[0])[0:n_digits]+' '+str(dark_bb[1])[0:n_digits]+' '+str(dark_bb[2])[0:n_digits]+' '+str(dark_bb[3])[0:n_digits] + '\n'
                vals = [int(s) if s.isdigit() else float(s) for s in line.split()]
                classno = vals[0]
                item_dict['category'] = tamara_berg_categories[classno]
                bb = [vals[1],vals[2],vals[3],vals[4]]
                print('classno {} ({}) bb {} imfile {} n_boxes {}'.format(classno,item_dict['category'],bb,imgfile,n_boxes))
#                bb = convert_dark_to_xywh((w,h),dark_bb)
                item_dict['bb'] = bb
        #         if use_visual_output:
        #             cv2.rectangle(img_arr, (bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),color=[int(255.0/10*classno),100,100],thickness=10)
        #             #resize to avoid giant images
        #             dest_height = 200
        #             dest_width = int(float(dest_height)/h*w)
        # #            print('h {} w{} destw {} desth {}'.format(h,w,dest_width,dest_height))
        #             factor = float(h)/dest_width
        #             newx = int(bb[0]*factor)
        #             newy = int(bb[0]*factor)
        #             im2 = cv2.resize(img_arr,(dest_width,dest_height))
        #             cv2.putText(im2,tamara_berg_categories[classno], (newx+1,newy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [int(255.0/10*classno),100,100],3)
        #
        #             cv2.imshow(imgfile,im2)
        #             cv2.waitKey(100)
                items.append(item_dict)
            # cv2.destroyAllWindows()
        #fp.close()
        info_dict['items'] = items
        print('db entry:'+str(info_dict))
        ack = db.training_images.insert_one(info_dict)
        print('ack:'+str(ack.acknowledged))

'''   db.training_images.find_one_and_update({'person_id': person_id},
                                              {'$set': {'gender': gender, 'status': 'done'}})
    image = db.genderator.find_one_and_update({'status': 'fresh'},
                                                  {'$set' : {'status': 'busy'}},
                                                  return_document=pymongo.ReturnDocument.AFTER)
'''
