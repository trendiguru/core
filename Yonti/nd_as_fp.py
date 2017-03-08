from ..constants import db
from ..paperdoll.neurodoll_falcon_client import nd
from tqdm import tqdm


def create_test_collection(category):
    new_col = 'nd_as_fp'
    all_items_in_category = db.shopstyle_US_Female.find({'categories':category})
    for item in tqdm(all_items_in_category):
        try:
            new_obj = {'img_url':item['images']['XLarge'],
                       'fp': item['fp'],
                       'category': category,
                       'nd': []}
            data = nd(new_obj['img_url'], get_layer_output='fc7')
            if not data['success']:
                raise Exception('nd failed')
            new_obj['nd'] = data['layer_output'].tolist()
            if not len(new_obj['nd']):
                raise Exception('empty array')

            db[new_col].insert_one(new_obj)
        except Exception as e:
            print e

