from ..constants import db
from ..paperdoll.neurodoll_falcon_client import nd
from tqdm import tqdm
import sys
import os








def create_test_collection(category):
    new_col = 'nd_as_fp'
    all_items_in_category = db.shopstyle_US_Female.find({'categories': category})
    for item in tqdm(all_items_in_category):
        sys.stdout = open(os.devnull, "w")
        try:
            new_obj = {'images': {'XLarge': item['images']['XLarge']},
                       'fp': item['fingerprint'],
                       'category': category,
                       'nd': []}
            data = nd(new_obj['images']['XLarge'], get_layer_output='fc7')
            if not data['success']:
                raise Exception('nd failed')
            new_obj['nd'] = data['layer_output'].tolist()
            if not len(new_obj['nd']):
                raise Exception('empty array')

            db[new_col].insert_one(new_obj)
        except Exception as e:
            print e

        finally:
            sys.stdout = sys.__stdout__