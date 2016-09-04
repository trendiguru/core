from __future__ import print_function
import xmltodict
from rq import Queue
from datetime import datetime
from ...Utils import get_cv2_img_array
from ...constants import db, redis_conn, fingerprint_version
from ..general.db_utils import print_error, get_hash, get_p_hash
from ...fingerprint_core import generate_mask_and_insert
from time import sleep
import re
today_date = str(datetime.date(datetime.now()))

q = Queue('fingerprinter4db', connection=redis_conn)

germender = {'herrenmode': 'Male',
             'damenmode': 'Female'}

cat2id_female = {'stockings': ['wetlook str', 'strapsst', 'mpfe', 'overknees'],  # before carigan
                 'leggings': ['leggings'],
                 'dress': ['kleid', 'petticoat', 'mididirndl', ],  # if not tock
                 'vest': ['westren'],
                 # check before kapuze,suit after jacket and cardigan # delete Jacquardwesten with yoga in imgurl and Hochzeitsanzug
                 'blazer': ['sakko', 'blazer'],
                 'jacket': ['jacke'],
                 'pants': ['hose'],  # but not strumpfhose no trachten!!
                 # check for shorts+swimsuit first # delete-> Lederhose,Trachten-Lederhosen,Kniebundhosen
                 't-shirt': ['t-shirt', 'tees'],
                 'sweater': ['sweater', 'sweats', 'pullunder', 'pullover', 'poncho'],
                 # first check for t-shirt,dress,sweatshirt
                 'sweatshirt': ['kapuzen', 'hoody'],
                 'cardigan': ['cardigans', 'strickwesten'],  # after coat
                 'shirt': ['trikot', 'shirt', 'hemd', 'kragen', 'oberteile'],  # but not still after sweater...
                 'suit': ['anz', 'blaumann', 'abendmode', 'overall'],
                 'tights': ['tights'],
                 'tanktop': ['sporttop', 'tank', 'funktionstop'],
                 'shorts': ['shorts', 'fitnesshose'],
                 'top': ['top'],
                 'coat': ['mantel', 'windstopper', 'bekleidung', 'cabanjacke', 'parka', 'Gehrock', ],
                 # check for shirt  before this
                 'jeans': ['jeans'],
                 'swimsuit': ['bademode'],
                 'skirt': ['dirndlsch', 'rock', 'skort'],  # check for coat before
                 'blouse': ['blusen'],
                 'bikini': ['bikinis']}  # but not shirt no trachten!! no Nacht and Lingerie

cat2id_male = {'vest': ['westren'],
               'blazer': ['sakko'],
               'jacket': ['jacke'],
               'pants': ['hose'],
               # check for shorts+swimsuit first # delete-> Lederhose,Trachten-Lederhosen,Kniebundhosen
               't-shirt': ['t-shirt', 'tees'],
               'sweater': ['sweater', 'sweats', 'pullunder', 'pullover'],
               'sweatshirt': ['kapuzen', 'hoody'],
               'cardigan': ['cardigans', 'strickwesten'],
               'shirt': ['trikot', 'shirt', 'hemden', 'kragen', 'oberteile'],
               'suit': ['anz', 'blaumann', 'abendmode'],
               'tights': ['tights'],
               'tanktop': ['sporttop', 'tank'],
               'shorts': ['shorts', 'fitnesshose'],
               'top': ['top'],
               'coat': ['mantel', 'windstopper', 'bekleidung'],  # check for shirt before this
               'jeans': ['jeans'],
               'swimsuit': ['bademode']}  # but not shirt


def match_female_cat(cats):
    for cat in cats:
        if any(seq for seq in ['trachten', 'nacht', 'lingerie', 'strumpfhose', 'hochzeitsanzug'] if seq in cat):
            return None

    female_keys = cat2id_female.keys()
    length = len(cats)
    category = None
    for i in range(length):
        candidate = cats[length - 1 - i]
        for key in female_keys:
            if any(seq for seq in cat2id_female[key] if seq in candidate):
                category = key
                if category == 'pants':
                    if any(cat for cat in cats if 'shorts' in cat):
                        return 'shorts'
                    elif cats[0] == 'bademode':
                        return 'swimsuit'
                    elif any(cat for cat in cats if cat in ['lederhose', 'tachten-lederhosen', 'kniebundhosen']):
                        return None
                    else:
                        return category
                elif category == 'skirt' or category == 'cardigan':
                    if any(cat for cat in cats if 'mantel' in cat):
                        return 'coat'
                    else:
                        return category
                elif category == 'shirt':
                    if any(cat for cat in cats if 'sweat' in cat):
                        return 'sweater'
                    elif any(cat for cat in cats if 'pull' in cat):
                        return 'sweater'
                    else:
                        return category
                elif category == 'vest':
                    if any(cat for cat in cats if 'jacke' in cat):
                        return 'jacket'
                    elif any(cat for cat in cats if 'cardigan' in cat):
                        return 'cardigan'
                    elif 'jacquardwesten' in cats:
                        return None
                    else:
                        return category
                else:
                    return category
    return category


def match_male_cat(cats):
    male_keys = cat2id_male.keys()
    length = len(cats)
    category = None
    for i in range(length):
        candidate = cats[length -1 -i]
        for key in male_keys:
            if any(seq for seq in cat2id_male[key] if seq in candidate):
                category = key
                if category == 'pants':
                    if any(cat for cat in cats if 'shorts' in cat):
                        return 'shorts'
                    elif cats[0]=='bademode':
                        return 'swimsuit'
                    elif any(cat for cat in cats if cat in ['lederhose', 'tachten-lederhosen', 'kniebundhosen']):
                        return None
                    else:
                        return category
                else:
                    return category
    return category


def find_category(family_tree):
    family_tree_lower = family_tree.lower()
    split_tree = re.split(r' > ', family_tree_lower)
    top_cat = split_tree[0]
    if top_cat != 'mode':
        return None, None
    sec_cat = split_tree[1]
    if sec_cat not in germender.keys():
        return None, None
    gender = germender[split_tree[1]]
    if gender == 'Male':
        return gender, match_male_cat(split_tree[2:])
    elif gender == 'Female':
        return gender, match_female_cat(split_tree[2:])
    else:
        return None, None


def process_xml(file_name):
    f = open(file_name)
    d = xmltodict.parse(f)
    d1 = d['Products']
    d2 = d1['Product']
    total_items = len(d2)
    one_percent = int(total_items/100)
    print ('[', end='')
    tmp = 0
    cat_ids = []
    cats = []
    for x, item in enumerate(d2):
        if divmod(x, one_percent)[0] > tmp:
            print ('#', end='')
            tmp += 1

        item_id = item['@ArticleNumber']
        #todo if...
        family_tree = item['CategoryPath']['ProductCategoryPath']
        gender, category = find_category(family_tree)
        if gender is None or category is None:
            continue

        if gender == 'Male' :
            collection_name = 'xl_Male'
        else:
            collection_name = 'xl_Female'

        category_id = item['CategoryPath']['ProductCategoryID']

        price = {'price': item['Price']['Price'],
                 'currency': item['Price']['CurrencySymbol'],
                 'priceLabel': item['Price']['DisplayPrice']}

        click_url = item['Deeplinks']['Product']

        image_url = item['Images']['Img'][0]['URL']

        image = get_cv2_img_array(image_url)
        if image is None:
            continue

        short_d = item['Details']['DescriptionShort']
        long_d = item['Details']['Description']
        brand = item['Details']['Brand']
        title = item['Details']['Title']

        new_item = {'id': item_id,
                    'category_id': category_id,
                    'clickUrl': click_url,
                    'images': {'XLarge': image_url},
                    'price': price,
                    'shortDescription': short_d,
                    'longDescription': long_d,
                    'brand': brand,
                    'tree': family_tree,
                    'gender': gender,
                    'status': {'instock': True, 'days_out': 0},
                    'fingerprint': {},
                    # 'img_hash': img_hash,
                    # 'p_hash': p_hash,
                    'download_data': {'dl_version': today_date,
                                      'first_dl': today_date,
                                      'fp_version': fingerprint_version},
                    'categories': category,
                    'raw_info': item}

        while q.count > 5000:
            sleep(30)

        q.enqueue(generate_mask_and_insert, args=(new_item, image_url, today_date, collection_name, image, False),
                  timeout=1800)

    print ('download finished')

if __name__ == "__main__":
    db.xl_raw.delete_many({})
    filename = '/home/developer/yonti/affilinet_products_5420_777756.xml'
    process_xml(filename)


