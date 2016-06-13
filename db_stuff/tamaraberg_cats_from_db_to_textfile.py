__author__ = 'jeremy'
import numpy as np
import os
import cv2

from trendi import constants

def cats_from_db(image_dir='/home/jeremy/image_dbs/tamara_berg/images'):
    db = constants.db
    cursor = db.training_images.find({'already_done':True})
    n_done = cursor.count()
    print(str(n_done)+' docs done')
    with open('tb_cats_from_webtool.txt','w') as fp:
        for i in range(n_done):
            document = cursor.next()
            url = document['url']
            filename = os.path.basename(url)
            full_path = os.path.join(image_dir,filename)
            items_list = document['items'] #
            hotlist = np.zeros(len(constants.web_tool_categories))
            for item in items_list:
                cat = item['category']
                if cat in constants.web_tool_categories:
                    index = constants.web_tool_categories.index(cat)
                else:
                    if cat in constants.tamara_berg_to_web_tool_dict:
                        cat = constants.tamara_berg_to_web_tool_dict[cat]
                        index = constants.web_tool_categories.index(cat)
                hotlist[index] = 1
                print('item:'+str(cat))
            print('hotlist:'+str(hotlist))
            line = str(full_path) +' '+ ' '.join(str(int(n)) for n in hotlist)
            fp.write(line+'\n')

def inspect_textfile(filename = 'tb_cats_from_webtool.txt'):
    with open(filename,'r') as fp:
        for line in fp:
            print line
            path = line.split()[0]
            cats = ''
            for i in range(len(constants.web_tool_categories)):
                current_val = int(line.split()[i+1])
                print('cur digit {} val {}'.format(i,current_val))
                if current_val:
                    cats = cats + constants.web_tool_categories[i]
                    print(cats)
            img_arr = cv2.imread(path)
            cv2.imshow('file',img_arr)
            cv2.waitKey(0)

if __name__ == "__main__": #
    cats_from_db()
    inspect_textfile()