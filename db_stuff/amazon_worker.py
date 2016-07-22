from ..constants import db, redis_conn, fingerprint_version
from ..Utils import get_cv2_img_array
from rq import Queue
from datetime import datetime
from ..fingerprint_core import generate_mask_and_insert
from time import sleep
import re
from db_utils import print_error, get_hash, categories_keywords, categories_swap

today_date = str(datetime.date(datetime.now()))

q = Queue('new_collection_fp', connection=redis_conn)

plus_sizes = ['XL', '1X', '2X', '3X', '4X', 'X', 'XX', 'XXX', 'XXXX', 'XXXXX', 'LARGE', 'PLUS']


def find_paperdoll_cat(category, short_desc, long_desc):
    desc = '%s,%s' % (short_desc, long_desc)
    DESC = desc.upper()
    all_possible_relevant_cats = re.split(r' |-|,|;|:|\.', DESC)
    all_possible_relevant_cats.append(category)

    categories = []
    for cat in all_possible_relevant_cats:
        if cat in categories_keywords:
            relevant_cat = categories_swap[cat]
            categories.append(relevant_cat)

    if len(categories) < 1:
        return ''

    if len(categories) > 1:
        if all(x for x in ['dress', 'shirt'] if x in categories):
            return 'shirt'
        if all(x for x in ['suit', 'shirt'] if x in categories):
            return 'shirt'
        if all(x for x in ['dress', 'pants'] if x in categories):
            return 'pants'
        if all(x for x in ['suit', 'pants'] if x in categories):
            return 'pants'

        if 'dress' in categories:
            return 'dress'

        if 'bikini' in categories:
            return 'bikini'

        if 'swimsuit' in categories:
            return 'swimsuit'

        if 'shirt' in categories:
            return 'shirt'

    category = categories[0]
    return category


def verify_plus_size(size_list):
    splited_list = []
    for size in size_list:
        size_upper = size.upper()
        split = re.split(r'\(|\)| |-|,', size_upper)
        for s in split:
            splited_list.append(s)
    if 'SMALL' in splited_list:
        return False
    return any(size for size in splited_list if size in plus_sizes)


def insert_items(collection_name, item_list, items_in_page, print_flag, family_tree, plus_size_flag=False):
    collection = db[collection_name]

    col_name_parts= re.split(r'_', collection_name)
    gender = col_name_parts[-1]
    new_items_count = 0
    for x, item in enumerate(item_list):
        if (x + 1) > items_in_page:
            break

        try:
            item_keys = item.keys()
            if any(x for x in ['ASIN', 'DetailPageURL', 'OfferSummary', 'ItemAttributes'] if x not in item_keys):
                if print_flag:
                    print_error('%s not in item keys' % x)
                continue

            asin = item['ASIN']
            asin_exists = collection.find_one({'asin': asin})
            if asin_exists:
                if print_flag:
                    print_error('item exists already!')
                continue

            if 'ParentASIN' not in item_keys:
                parent_asin = asin
            else:
                parent_asin = item['ParentASIN']

            click_url = item['DetailPageURL']
            offer = item['OfferSummary']['LowestNewPrice']
            price = {'price': float(offer['Amount']) / 100,
                     'currency': offer['CurrencyCode'],
                     'priceLabel': offer['FormattedPrice']}
            attributes = item['ItemAttributes']
            attr_keys = attributes.keys()
            if 'ClothingSize' in attr_keys:
                clothing_size = attributes['ClothingSize']
            elif 'Size' in attr_keys:
                clothing_size = attributes['Size']
            else:
                if print_flag:
                    print_error('No Size', attr_keys)
                continue

            if 'Brand' in attr_keys:
                brand = attributes['Brand']
            else:
                brand = 'unknown'

            if 'ProductTypeName' in attr_keys:
                tmp_category = attributes['ProductTypeName']
            else:
                tmp_category = '2BDtermind'

            color = attributes['Color']
            sizes = [clothing_size]
            if plus_size_flag:
                plus_size = verify_plus_size(sizes)
                if not plus_size:
                    if print_flag:
                        print_error('Not a Plus Size', attr_keys)
                    continue

            parent_asin_exists = collection.find_one({'parent_asin': parent_asin, 'features.color': color})
            if parent_asin_exists:
                sizes = parent_asin_exists['features']['sizes']
                if clothing_size not in sizes:
                    sizes.append(clothing_size)
                    collection.update_one({'_id': parent_asin_exists['_id']}, {'$set': {'features.sizes': sizes}})
                    if print_flag:
                        print_error('added another size to existing item')
                else:
                    if print_flag:
                        print_error('parent_asin + color + size already exists ----- %s->%s' %
                                    (color, clothing_size))
                continue

            if 'LargeImage' in item_keys:
                image_url = item['LargeImage']['URL']
            elif 'MediumImage' in item_keys:
                image_url = item['MediumImage']['URL']
            elif 'SmallImage' in item_keys:
                image_url = item['SmallImage']['URL']
            else:
                if print_flag:
                    print_error('No image')
                continue

            img_url_exists = collection.find_one({'images.XLarge': image_url})
            if img_url_exists:
                print ('img url already exists')
                continue

            image = get_cv2_img_array(image_url)
            if image is None:
                if print_flag:
                    print_error('bad img url')
                continue

            img_hash = get_hash(image)

            hash_exists = collection.find_one({'img_hash': img_hash})
            if hash_exists:
                print ('hash already exists')
                continue

            short_d = attributes['Title']
            if 'Feature' in attr_keys:
                long_d = ' '.join(attributes['Feature'])
            else:
                long_d = ''

            category = find_paperdoll_cat(tmp_category, short_d, long_d)
            if len(category)==0:
                category='unKnown'

            new_item = {'asin': asin,
                        'parent_asin': parent_asin,
                        'clickUrl': click_url,
                        'images': {'XLarge': image_url},
                        'price': price,
                        'color': color,
                        'sizes': sizes,
                        'shortDescription': short_d,
                        'longDescription': long_d,
                        'brand': brand,
                        'categories': category,
                        'raw_info': attributes,
                        'tree': family_tree,
                        'status': {'instock': True, 'days_out': 0},
                        'fingerprint': None,
                        'gender': gender,
                        'img_hash': img_hash,
                        'download_data': {'dl_version': today_date,
                                          'first_dl': today_date,
                                          'fp_version': fingerprint_version},
                        }

            while q.count > 5000:
                sleep(30)

            q.enqueue(generate_mask_and_insert, args=(new_item, image_url, today_date, collection_name, image, False),
                      timeout=1800)

            new_items_count += 1

        except Exception as e:
            print_error('ERROR', e)
            pass

    print('%d new items inserted to %s' % (new_items_count, collection_name))
