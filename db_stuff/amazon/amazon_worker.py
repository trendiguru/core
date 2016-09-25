# -*- coding: utf-8 -*-

import re
from datetime import datetime
from time import sleep

from rq import Queue

from ...Utils import get_cv2_img_array
from ...constants import db, redis_conn, fingerprint_version
from ..general.db_utils import print_error, get_hash, get_p_hash
from ...fingerprint_core import generate_mask_and_insert

today_date = str(datetime.date(datetime.now()))

q = Queue('fingerprinter4db', connection=redis_conn)
pants = ['PANTS', 'PANT', 'TROUSERS', 'TROUSER', 'CULOTTE', 'CULOTTES', 'CHINO', 'CHINOS', 'CAPRI', 'CAPRIS', 'SLACKS',
         'PONTE']
big_no_no = ['PANTIES', 'BRIEFS', 'UNDERPANTS', 'UNDERWEAR', 'BOXER', 'PANTIE', 'BRIEF', 'CUFFLINK', 'STUDS',
             'KILT', 'STRAP']


def amazon_de_to_ppd(cat, sub_cat, title):
    if cat == u"Badeanzüge":
        return 'swimsuit'
    elif cat in ['Bikini - Sets', 'Bikinioberteile']:
        return 'bikini'
    elif cat in ['Pareos & Strandkleider', 'Kleider']:
        return 'dress'
    elif cat in ['Badeshorts', 'Tankinis', 'Shorts', 'Badehosen']:
        return 'shorts'
    elif cat in ['Shirts & Blusen', 'Blusen & Tuniken']:
        return 'blouse'
    elif cat in ['Hosen', 'Smokinghosen', 'Anzughosen']:
        return 'pants'
    elif cat in ['Jacken', 'Smokingjacken', 'Sakkos', 'Anzugjacken']:
        return 'jacket'
    elif cat == u"Mäntel":
        return 'coat'
    elif cat == 'Westen':
        return 'vest'
    elif cat == 'Jeanshosen':
        return 'jeans'
    elif cat in ['Jumpsuits', u"Anzüge", 'Smokings', 'Smoking & Frack']:
        return 'suit'
    elif cat in ['Boleros', 'Strickjacken', 'Strickwesten']:
        return 'cardigan'
    elif cat in ['Pullunder', 'Pullover']:
        return 'sweater'
    elif cat in ['Kapuzenpullover', 'Sweatshirts', 'Kappen']:
        return 'sweatshirt'
    elif cat in ['Tops', 'Oberteile']:
        return 'top'
    elif cat == 'T-Shirts':
        return 't-shirt'
    elif cat in ['Langarmshirts', 'Poloshirts', 'Shirts & Hemden']:
        return 'shirt'
    elif cat == u"Röcke":
        return 'skirt'
    elif cat == 'Leggings':
        return 'leggings'
    elif cat == 'Strumpfhosen & Leggings':
        return 'tights'
    else:
        return ''


def verify_by_title(title):
    title_upper = title.upper()
    if any(x in title_upper for x in big_no_no):
        return ''
    if any(x in title_upper for x in ['BLAZER', 'BLAZERS']):
        return 'blazer'
    if any(x in title_upper for x in ['STOCKING', 'STOCKINGS']):
        return 'stockings'
    elif any(x in title_upper for x in pants):
        if any(x in title_upper for x in ['SHORTS','SHORT']):
            return 'shorts'
        return 'pants'
    elif any(x in title_upper for x in ['DRESS', 'DRESSES', 'MAXI', 'GOWN']):
        return 'dress'
    elif any(x in title_upper for x in ['SHIRT', 'SHIRTS']):
        return 'shirt'
    elif any(x in title_upper for x in ['SUIT', 'TOXEDO']):
        return 'suit'
    elif any(x in title_upper for x in ['SWEATSHIRT', 'SWEATSHIRTS']):
        return 'sweatshirt'
    elif any(x in title_upper for x in ['SWEATER', 'SWEATERS']):
        return 'sweater'
    elif 'SHORTS' in title_upper:
        return 'shorts'
    elif any(x in title_upper for x in ['T-SHIRT', 'T-SHIRTS']):
        return 't-shirt'
    elif any(x in title_upper for x in ['SKIRT', 'SKIRTS', 'SKORT', 'SKORTS', 'MINI']):
        return 'skirt'
    elif any(x in title_upper for x in ['COAT', 'FAUX', 'COATS', 'OUTWEAR', 'PARKA', 'PARKAS']):
        return 'coat'
    elif any(x in title_upper for x in ['JACKET', 'JACKETS']):
        return 'jacket'
    elif 'TIGHTS' in title_upper:
        return 'tights'
    elif any(x in title_upper for x in ['TOP', 'TOPS']):
        return 'top'
    elif 'JEANS' in title_upper:
        return 'jeans'
    else:
        return ''


def amazon_usa_to_ppd(cat, sub_cat, title):
    if cat == 'Dresses':
        return 'dress'
    if cat == 'tights':
        return verify_by_title(title)
    if cat == 'stockings':
        return 'stockings'
    elif cat == 'Tops & Tees':
        if sub_cat == 'Blouses & Button-Down Shirts':
            return 'blouse'
        elif sub_cat == 'Henleys':
            return 'sweater'
        elif sub_cat == 'Knits & Tees':
            return 't-shirt'
        elif sub_cat == 'Polos':
            return 'shirt'
        elif sub_cat == 'Tanks & Camis':
            return 'tanktop'
        elif sub_cat == 'Tunics':
            return 'top'
        elif sub_cat == 'Vests':
            return 'vest'
        else:
            return 'top'
    elif cat == 'Sweaters':
        if sub_cat == 'Cardigans':
            return 'cardigan'
        elif sub_cat == 'Pullovers':
            return 'sweatshirt'
        elif sub_cat == 'Shrugs':
            return 'sweater'
        elif sub_cat == 'Vests':
            return 'vest'
        else:
            return 'sweater'
    elif cat == 'Fashion Hoodies & Sweatshirts':
        return 'sweatshirt'
    elif cat == 'Jeans':
        return 'jeans'
    elif cat == 'Pants':
        return 'pants'
    elif cat == 'Skirts':
        return 'skirt'
    elif cat == 'Shorts':
        return 'shorts'
    elif cat == 'Leggings':
        return 'leggings'
    elif cat == 'Active':
        if sub_cat == 'Active Hoodies' or sub_cat == 'Active Sweatshirts':
            return 'sweatshirt'
        elif 'Track & Active Jackets':
            return verify_by_title(title)
        elif sub_cat == 'Active Top & Bottom Sets':
            return ''
        elif sub_cat == 'Active Shirts & Tees':
            return 'shirt'
        elif sub_cat == 'Active Pants':
            return 'pants'
        elif sub_cat == 'Active Leggings':
            return 'leggings'
        elif sub_cat == 'Active Shorts':
            return 'shorts'
        elif sub_cat == 'Active Skirts' or sub_cat == 'Active Skorts':
            return 'skirt'
        else:
            return ''
    elif cat == 'Swimsuits & Cover Ups':
        if sub_cat == 'Bikinis' or sub_cat == 'Tankini':
            return 'bikini'
        else:
            return 'swimsuit'
    elif cat == 'Jumpsuits, Rompers & Overalls':
        return 'roampers'
    elif cat == 'Coats, Jackets & Vests':
        if sub_cat in ['Down & Parkas', 'Wool & Pea Coats', 'Fur & Faux Fur']:
            return 'coat'
        elif sub_cat in ['Denim Jackets', 'Quilted Lightweight Jackets', 'Casual Jackets', 'Leather & Faux Leather']:
            return 'jacket'
        elif sub_cat == 'Vests':
            return 'vest'
        else:
            return ''
    elif cat == 'Suiting & Blazers':
        if sub_cat == 'Blazers':
            return 'blazer'
        elif sub_cat == 'Separates':
            return verify_by_title(title)
        else:
            return 'suit'
    elif cat == 'Shirts':
        if sub_cat == 'T-Shirts':
            return 't-shirt'
        elif sub_cat == 'Casual Button-Down Shirts':
            return 'shirt'
        elif sub_cat == 'Tank Tops':
            return 'tanktop'
        elif sub_cat == 'Henleys':
            return 'sweater'
        elif sub_cat == 'Polos':
            return 'shirt'
        else:
            return 'top'
    elif cat == 'Jackets & Coats':
        if sub_cat in ['Down & Down Alternative', 'Outerwear', 'Trench & Rain', 'Wool & Blends']:
            return 'coat'
        elif sub_cat in ['Fleece', 'Leather & Faux Leather', 'Lightweight Jackets']:
            return 'jacket'
        elif sub_cat == 'Vests':
            return 'vest'
        else:
            return ''
    elif cat == 'Swim':
        return 'swimsuit'
    elif cat == 'Suits & Sport Coats':
        if sub_cat in ['Suits', 'Tuxedos']:
            return verify_by_title(title)
        elif sub_cat == 'Suit Separates':
            return verify_by_title(title)
        elif sub_cat == 'Sport Coats & Blazers':
            return 'blazer'
        elif sub_cat == 'Vests':
            return 'vest'
        else:
            return ''
    else:
        return ''


def find_paperdoll_cat(family, title, cc):
    leafs = re.split(r'->', family)
    if cc == 'DE':
        leaf_length = len(leafs)
        category = leafs[leaf_length-1]
    else:
        category = leafs[3]
    sub_category = None
    sub1 = None
    if len(leafs) > 4:
        sub1 = leafs[4]
        sub_category = '%s.%s' % (category, sub1)
    if len(leafs) > 5:
        sub2 = leafs[5]
        sub_category = '%s.%s' % (sub_category, sub2)
    if cc == 'US':
        category = amazon_usa_to_ppd(category, sub1, title)
    elif cc == 'DE':
        category = amazon_de_to_ppd(category, sub1, title)

    return category, sub_category


def update_dl_version(current_version, asin, collection_name):
    if current_version != today_date:
        collection = db[collection_name]
        collection.update_one({'id': asin}, {'$set': {'download_data.dl_version': today_date}})


def insert_items(collection_name, cc, item_list, items_in_page, print_flag, family_tree):
    collection = db[collection_name]

    col_name_parts = re.split(r'_', collection_name)
    gender = col_name_parts[-1]
    new_items_count = 0
    for x, item in enumerate(item_list):
        if (x + 1) > items_in_page:
            break

        try:
            item_keys = item.keys()
            if any(x for x in ['ASIN', 'DetailPageURL', 'OfferSummary', 'ItemAttributes', 'ProductGroup'] if x not in item_keys):
                if print_flag:
                    print_error('%s not in item keys' % x)
                continue
            group = item['ProductGroup']
            if group != 'Apparel' and group != 'Sports':
                print('########################################################################')
                continue
            asin = item['ASIN']
            asin_exists = collection.find_one({'id': asin})
            if asin_exists:
                if print_flag:
                    print_error('item exists already!')
                update_dl_version(asin_exists['download_data']['dl_version'], asin, collection_name)
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
            if 'Brand' in attr_keys:
                brand = attributes['Brand']
            else:
                brand = 'unknown'

            if cc == 'US':
                if 'ClothingSize' in attr_keys:
                    clothing_size = attributes['ClothingSize']
                elif 'Size' in attr_keys:
                    clothing_size = attributes['Size']
                else:
                    if print_flag:
                        print_error('No Size', attr_keys)
                    continue

                color = attributes['Color']
                sizes = [clothing_size]
                parent_asin_exists = collection.find_one({'parent_asin': parent_asin, 'features.color': color})
                if parent_asin_exists:
                    sizes = parent_asin_exists['features']['sizes']
                    update_dl_version(parent_asin_exists['download_data']['dl_version'], asin, collection_name)
                    if clothing_size not in sizes:
                        sizes.append(clothing_size)
                        collection.update_one({'_id': parent_asin_exists['_id']}, {'$set': {'features.sizes': sizes}})
                        if print_flag:
                            print_error('added another size to existing item')
                    else:
                        if print_flag:
                            print_error('parent_asin + color + size already exists ----- %s->%s' %
                                        (color, clothing_size))
                        update_dl_version(parent_asin_exists['download_data']['dl_version'], asin, collection_name)
                    continue

            else:
                color = None
                sizes = []
                parent_asin_exists = collection.find_one({'parent_asin': parent_asin})
                if parent_asin_exists:
                    update_dl_version(parent_asin_exists['download_data']['dl_version'], asin, collection_name)
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
                update_dl_version(img_url_exists['download_data']['dl_version'], asin, collection_name)
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
                update_dl_version(hash_exists['download_data']['dl_version'], asin, collection_name)
                continue

            p_hash = get_p_hash(image)
            p_hash_exists = collection.find_one({'p_hash': p_hash})
            if p_hash_exists:
                print ('p_hash already exists')
                update_dl_version(p_hash_exists['download_data']['dl_version'], asin, collection_name)
                continue

            short_d = attributes['Title']
            if 'Feature' in attr_keys:
                long_d = ' '.join(attributes['Feature'])
            else:
                long_d = ''

            category, sub_category = find_paperdoll_cat(family_tree, short_d, cc)
            if len(category) == 0:
                category = 'unknown'

            new_item = {'id': asin,
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
                        'sub_category': sub_category,
                        'raw_info': attributes,
                        'tree': family_tree,
                        'status': {'instock': True, 'days_out': 0},
                        'fingerprint': {},
                        'gender': gender,
                        'img_hash': img_hash,
                        'p_hash': p_hash,
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


