__author__ = 'jeremy'
import numpy as np
from trendi import constants

def cats_from_db():
    db = constants.db
    cursor = db.training_images.find({'already_done':True})
    n_done = cursor.count()
    print(str(n_done)+' docs done')
    for i in range(n_done):
        document = cursor.next()
        url = document['url']
        items_list = document['items']
        hotlist = np.zeros(len(constants.web_tool_categories))
        for item in items_list:
            cat = item['category']
            index = constants.web_tool_categories.index(cat)
            hotlist[index] = 1
            print('item:'+str(cat))
        print('hotlist:'+str(hotlist))

if __name__ == "__main__":
    cats_from_db()