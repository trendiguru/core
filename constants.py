import os

import cv2
import pymongo
from redis import Redis


# file containing constants for general TG use
# fingerprint related consts

fingerprint_length = 696
fingerprint_version = 792015  # DayMonthYear
extras_length = 6
histograms_length = [180, 255, 255]
fingerprint_weights = [0.05, 0.5, 0.225, 0.225]
K = 0.5                     # for euclidean distance
min_bb_to_image_area_ratio = 0.95  # if bb takes more than this fraction of image area then use  cv2.GC_INIT_WITH_RECT instead of init with mask


##############
# rq / worker stuff
##############

pd_worker_command =  'cd /home/jeremy/paperdoll3/paperdoll-v1.0/ && /usr/bin/python /usr/local/bin/rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd'
N_expected_pd_workers_per_server = 15


#########
# DB stuff
#########

parallel_matlab_queuename = 'pd'
nonparallel_matlab_queuename = 'pd_nonparallel'
caffe_path_in_container = '/opt/caffe'
# db = pymongo.MongoClient(host=os.environ["MONGO_HOST"], port=int(os.environ["MONGO_PORT"])).mydb
# redis_conn = Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
db = pymongo.MongoClient(host="mongodb1-instance-1").mydb
redis_conn = Redis(host="redis1-redis-1-vm")
# new worker : rqworker -u redis://redis1-redis-1-vm:6379 [name] &
redis_conn_old = Redis()
update_collection_name = 'products'

############
## caffe stuff
############

caffeRelevantLabels = [601, 608, 610, 614, 617, 638, 639, 655, 689, 697, 735, 775, 834, 841, 264, 401, 400]

# fp rating related constants
min_image_area = 400
min_images_per_doc = 10  # item has to have at least this number of pics
max_images_per_doc = 18  # item has to have less than this number of pics
max_items = 50  # max number of items to consider for rating fingerprint

project_dir = os.path.dirname(__file__)
classifiers_folder = os.path.join(project_dir, 'classifiers')
# classifiers_folder = "/home/ubuntu/Dev/trendi_guru_modules/classifiers/"

# classifier to category relation
classifier_to_category_dict = {"dressClassifier.xml": ["dresses", "bridal-mother-dresses", "bridal-bridesmaid-dresses",
                                                       "maternity-dresses"],
                               "pantsClassifier.xml": ["womens-pants", "mens-athletic-pants", "teen-guys-pants",
                                                       "mens-pants", "mens-jeans", "mens-big-and-tall-jeans",
                                                       "mens-big-and-tall-pants", "teen-girls-pants", "womens-pants",
                                                       "athletic-pants", "jeans", "plus-size-pants", "plus-size-jeans",
                                                       "petite-pants", "petite-jeans", "maternity-pants",
                                                       "maternity-jeans"],
                               "shirtClassifier.xml": ["womens-tops", "teen-girls-sweaters", "teen-girls-tops",
                                                       "teen-girls-outerwear", "teen-girls-jackets",
                                                       "teen-girls-sweatshirts", "athletic-tops", "athletic-jackets",
                                                       "sweatshirts", "plus-size-outerwear", "plus-size-sweatshirts",
                                                       "plus-size-sweaters", "plus-size-jackets", "plus-size-tops",
                                                       "jackets", "petite-sweatshirts", "petite-outerwear",
                                                       "petite-sweaters", "petite-jackets", "petite-tops",
                                                       "maternity-tops", "maternity-sweaters", "maternity-jackets",
                                                       "maternity-outerwear", "sweaters", "womens-outerwear",
                                                       "mens-athletic-jackets", "mens-athletic-shirts",
                                                       "teen-guys-outerwear", "teen-guys-shirts", "teen-guys-blazers",
                                                       "teen-guys-sweaters", "mens-tee-shirts", "mens-shirts",
                                                       "mens-outerwear", "mens-sweatshirts", "mens-sweaters",
                                                       "mens-big-and-tall-shirts", "mens-big-and-tall-sweaters",
                                                       "mens-big-and-tall-coats-and-jackets",
                                                       "mens-big-and-tall-blazers"]}

db_relevant_items = ['women', 'womens-clothes', 'womens-suits', 'shorts', 'blazers', 'tees-and-tshirts',
                     'jeans', 'bootcut-jeans', 'classic-jeans', 'cropped-jeans', 'distressed-jeans',
                     'flare-jeans', 'relaxed-jeans', 'skinny-jeans', 'straight-leg-jeans', 'stretch-jeans',
                     'womens-tops', 'button-front-tops', 'camisole-tops', 'cashmere-tops', 'halter-tops',
                     'longsleeve-tops', 'shortsleeve-tops', 'sleeveless-tops', 'tank-tops', 'tunic-tops', 'polo-tops',
                     'skirts', 'mini-skirts', 'mid-length-skirts', 'long-skirts',
                     'sweaters', 'sweatshirts', 'cashmere-sweaters', 'cardigan-sweaters', 'crewneck-sweaters',
                     'turleneck-sweaters', 'v-neck-sweaters',
                     'womens-pants', 'wide-leg-pants', 'skinny-pants', 'dress-pants', 'cropped-pants', 'casual-pants',
                     'dresses', 'cocktail-dresses', 'day-dresses', 'evening-dresses',
                     'jackets', 'casual-jackets', 'leather-jackets', 'vests',
                     'coats', 'womens-outerwear', 'fur-and-shearling-coats', 'leather-and-suede-coats',
                     'puffer-coats', 'raincoats-and-trenchcoats', 'wool-coats',
                     'leggings']

# paperdoll items' legends

paperdoll_shopstyle_women = {'top': 'womens-tops', 'pants': 'womens-pants', 'shorts': 'shorts', 'jeans': 'jeans',
                             'jacket': 'jackets', 'blazer': 'blazers', 'shirt': 'womens-tops', 'skirt': 'skirts',
                             'blouse': 'womens-tops', 'dress': 'dresses', 'sweater': 'sweaters',
                             't-shirt': 'tees-and-tshirts', 'cardigan': 'cardigan-sweaters', 'coat': 'coats',
                             'suit': 'womens-suits', 'vest': 'vests', 'sweatshirt': 'sweatshirts',
                             'jumper': 'v-neck-sweaters', 'bodysuit': 'shapewear', 'leggings': 'leggings',
                             'stockings': 'hosiery', 'tights': 'leggings'}

paperdoll_shopstyle_women_jp_categories = {
    'blazer': {'id': u'\u30c6\u30fc\u30e9\u30fc\u30c9\u30b8\u30e3\u30b1\u30c3\u30c8',
               'name': u'\u30c6\u30fc\u30e9\u30fc\u30c9\u30b8\u30e3\u30b1\u30c3\u30c8'},
    'blouse': {'id': u'\u30d1\u30fc\u30c6\u30a3\u30c9\u30ec\u30b9',
               'name': u'\u30d1\u30fc\u30c6\u30a3\u30c9\u30ec\u30b9'},
    'bodysuit': {'id': u'\u30ec\u30ae\u30f3\u30b9',
                 'name': u'\u30ec\u30ae\u30f3\u30b9'},
    'cardigan': {'id': u'\u30ab\u30fc\u30c7\u30a3\u30ac\u30f3',
                 'name': u'\u30ab\u30fc\u30c7\u30a3\u30ac\u30f3'},
    'coat': {'id': u'\u30b3\u30fc\u30c8', 'name': u'\u30b3\u30fc\u30c8'},
    'dress': {'id': u'\u5927\u304d\u3044\u30b5\u30a4\u30ba-\u30b7\u30e7\u30fc\u30c8\u30d1\u30f3\u30c4',
              'name': u'\u5927\u304d\u3044\u30b5\u30a4\u30ba \u30b7\u30e7\u30fc\u30c8\u30d1\u30f3\u30c4'},
    'jacket': {'id': u'\u30b3\u30fc\u30c8', 'name': u'\u30b3\u30fc\u30c8'},
    'jeans': {'id': u'\u30ef\u30f3\u30d4\u30fc\u30b9',
              'name': u'\u30ef\u30f3\u30d4\u30fc\u30b9'},
    'jumper': {'id': u'v\u30cd\u30c3\u30af', 'name': u'V\u30cd\u30c3\u30af'},
    'leggings': {'id': u'\u30af\u30e9\u30b7\u30c3\u30af\u30c7\u30cb\u30e0',
                 'name': u'\u30af\u30e9\u30b7\u30c3\u30af\u30c7\u30cb\u30e0'},
    'pants': {'id': u'\u30af\u30e9\u30b7\u30c3\u30af\u30c7\u30cb\u30e0',
              'name': u'\u30af\u30e9\u30b7\u30c3\u30af\u30c7\u30cb\u30e0'},
    'shirt': {'id': u'\u30d1\u30fc\u30c6\u30a3\u30c9\u30ec\u30b9',
              'name': u'\u30d1\u30fc\u30c6\u30a3\u30c9\u30ec\u30b9'},
    'shorts': {'id': u'\u30d5\u30ec\u30a2\u30c7\u30cb\u30e0',
               'name': u'\u30d5\u30ec\u30a2\u30c7\u30cb\u30e0'},
    'skirt': {'id': u'\u30d5\u30ec\u30a2\u30c7\u30cb\u30e0',
              'name': u'\u30d5\u30ec\u30a2\u30c7\u30cb\u30e0'},
    'stockings': {'id': u'\u30af\u30e9\u30b7\u30c3\u30af\u30c7\u30cb\u30e0',
                  'name': u'\u30af\u30e9\u30b7\u30c3\u30af\u30c7\u30cb\u30e0'},
    'suit': {'id': u'\u30d1\u30fc\u30c6\u30a3\u30c9\u30ec\u30b9',
             'name': u'\u30d1\u30fc\u30c6\u30a3\u30c9\u30ec\u30b9'},
    'sweater': {'id': u'\u30ef\u30f3\u30d4\u30fc\u30b9',
                'name': u'\u30ef\u30f3\u30d4\u30fc\u30b9'},
    'sweatshirt': {'id': u'\u5927\u304d\u3044\u30b5\u30a4\u30ba-\u30c7\u30cb\u30e0',
                   'name': u'\u5927\u304d\u3044\u30b5\u30a4\u30ba \u30c7\u30cb\u30e0'},
    't-shirt': {'id': u't\u30b7\u30e3\u30c4', 'name': u'T\u30b7\u30e3\u30c4'},
    'tights': {'id': u'\u30af\u30e9\u30b7\u30c3\u30af\u30c7\u30cb\u30e0',
               'name': u'\u30af\u30e9\u30b7\u30c3\u30af\u30c7\u30cb\u30e0'},
    'top': {'id': u'\u30d1\u30fc\u30c6\u30a3\u30c9\u30ec\u30b9',
            'name': u'\u30d1\u30fc\u30c6\u30a3\u30c9\u30ec\u30b9'},
    'vest': {'id': u'\u30d9\u30b9\u30c8', 'name': u'\u30d9\u30b9\u30c8'}}

paperdoll_shopstyle_men = {'top': 'mens-shirts', 'pants': 'mens-pants', 'shorts': 'mens-shorts',
                           'jeans': 'mens-jeans', 'jacket': 'mens-outerwear', 'blazer': 'mens-outerwear',
                           'shirt': 'mens-shirts', 'skirt': 'mens-shorts', 'blouse': 'mens-shirts',
                           'dress': 'mens-suits', 'sweater': 'mens-sweaters', 't-shirt': 'mens-tees-and-tshirts',
                           'cardigan': 'mens-cardigan-sweaters', 'coat': 'mens-overcoats-and-trenchcoats',
                           'suit': 'mens-suits', 'vest': 'vests', 'sweatshirt': 'mens-sweatshirts',
                           'leggings': 'mens-pants', 'stockings': 'mens-pants', 'tights': 'mens-pants'}

paperdoll_categories = {"whole_body": ['bodysuit', 'dress', 'jumper', 'suit', 'romper'],
                        "upper_cover": ['blazer', 'cape', 'jacket', 'cardigan', 'coat', 'vest', 'sweatshirt'],
                        "upper_under": ['t-shirt', 'blouse', 'shirt', 'top', 'sweater', 'sweatshirt'],
                        "lower_cover": ['shorts', 'skirt', 'jeans', 'pants'],
                        "lower_under": ['stockings', 'tights', 'leggings']}

paperdoll_whole_body = ['bodysuit', 'dress', 'jumper', 'suit', 'romper', 'intimate']
paperdoll_upper = ['blazer', 'cape', 'jacket', 't-shirt', 'blouse', 'cardigan', 'shirt', 'coat', 'top', 'bra',
                   'sweater', 'vest', 'sweatshirt']
paperdoll_lower = ['pants', 'stockings', 'jeans', 'tights', 'leggings', 'shorts', 'skirt']
paperdoll_shoes = ['pumps', 'wedges', 'flats', 'clogs', 'shoes', 'boots', 'heels', 'loafers', 'sandals', 'sneakers']
paperdoll_accessories = ['tie', 'purse', 'hat', 'sunglasses', 'bag', 'belt']

nonlogic_clothing = [{'pants': ['jeans', 'stockings', 'jumper', 'suit', 'tights', 'leggings', 'shorts', 'romper',
                                'skirt', 'intimate']},
                     {'skirt': ['pants', 'jeans', 'shorts', 'romper', 'jumper']},
                     {'dress': ['t-shirt', 'blouse', 'jeans', 'shirt', 'bodysuit', 'jumper', 'suit',
                                'romper', 'shorts', 'top', 'skirt']},
                     {'jacket': ['blazer', 'cape', 'cardigan', 'sweater', 'sweatshirt', 'vest']}]

caffe_relevant_strings = ['hoopskirt', 'jean', 'blue_jean', 'denim', 'jersey', 'T-shirt', 'tee shirt', 'kimono',
                          'lab coat', 'tank suit', 'maillot', 'miniskirt', 'mini', 'overskirt', 'pajama', 'pyjama',
                          "pj's", 'jammies', 'poncho', 'sarong', 'suit', 'suit of clothes', 'sweatshirt']

flipkart_relevant_categories = ['Shirts', 'Skirts', 'Pants', 'Kurtas', 'Jackets', 'Dresses', 'Trousers', 'Kurtis',
                                'Leggings', 'Tunics', 'Tops', 'Coats', 'Shorts', 'Jumpsuits', 'Blazers', 'Cardigans',
                                'Sweaters', 'Sweatshirts', 'Jeans', 'Blouses', 'Salwars', 'Shrugs', 'Stockings',
                                'Shirt', 'Skirt', 'Pant', 'Jacket', 'Dress', 'Trouser',
                                'Legging', 'Tunic', 'Top', 'Coat', 'Short', 'Jumpsuit', 'Blazer', 'Cardigan',
                                'Sweater', 'Sweatshirt', 'Blouse', 'Suits', 'Tights', 'T-Shirt', 'T-Shirts']

flipkart_paperdoll_women = {'Tops': 'top', 'Top': 'top',
                            'Pants': 'pants', 'Pant': 'pants', 'Trousers': 'pants', 'Trouser': 'pants',
                            'Shorts': 'shorts', 'Short': 'shorts',
                            'Jeans': 'jeans',
                            'Jackets': 'jacket', 'Jacket': 'jacket',
                            'Blazers': 'blazer', 'Blazer': 'blazer',
                            'Shirts': 'shirt', 'Shirt': 'shirt',
                            'Skirts': 'skirt', 'Skirt': 'skirt',
                            'Tunics': 'blouse', 'Tunic': 'blouse', 'Blouses': 'blouse', 'Blouse': 'blouse',
                            'Dresses': 'dress', 'Dress': 'dress', 'Kurtis': 'dress', 'Kurtas': 'dress',
                            'Salwars': 'dress',
                            'Sweaters': 'sweater', 'Sweater': 'sweater',
                            'T-Shirts': 't-shirt', 'T-Shirt': 't-shirt',
                            'Cardigans': 'cardigan', 'Cardigan': 'cardigan', 'Shrugs': 'cardigan',
                            'Coats': 'coat', 'Coat': 'coat',
                            'Suits': 'suit', 'Jumpsuits': 'suit', 'Jumpsuit': 'suit',
                            'Leggings': 'tights', 'Tights': 'tights', 'Legging': 'tights',
                            'Sweatshirts': 'sweatshirt', 'Sweatshirt': 'sweater',
                            'Stockings': 'stockings'}

paperdoll_relevant_categories = {'top', 'pants', 'shorts', 'jeans', 'jacket', 'blazer', 'shirt', 'skirt', 'blouse',
                                 'dress',
                                 'sweater', 't-shirt', 'cardigan', 'coat', 'suit', 'tights', 'sweatshirt', 'stockings'}

# these are the fashionista db cats in order , e.g. the mask will have 1 for null (unknown) and 56 for skin
fashionista_categories = ['null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse','boots',
                          'blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings','scarf','hat',
                          'top','cardigan','accessories','vest','sunglasses','belt','socks','glasses','intimate',
                          'stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges','ring',
                          'flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch','pumps','wallet',
                          'bodysuit','loafers','hair','skin']

pd_output_savedir = '/home/jeremy/pd_output'
# for web bounding box interface
# this is for going to the previous item, highest numbered image
max_image_val = 666
svg_folder = '/var/www/static/svgs/'

svg_url_prefix = 'http://extremeli.trendi.guru/static/svgs/'

nadav = 'awesome'

Reserve_cpus = 2  # number of cpus to not use when doing stuff in parallel

# for gender id
gender_ttl = 5  # 10 seconds ttl , answer should be nearly immediate
paperdoll_ttl = 90  # seconds to wait for paperdoll result
caffe_general_ttl = 30  # seconds to wait for paperdoll result
general_ttl = 1000  # ttl of all queues

# QC worker voting params

# N_top_results - a list of N1, N2 etc where N1 items are shown to first wave of voters,
# then top N2 of those are shown to second wave of voters, etc

# the length of this list is the number of voting stages


N_top_results_to_show = [100, 20]
N_pics_per_worker = [5, 5]
N_workers = [N_top_results_to_show[0] / N_pics_per_worker[0], N_top_results_to_show[1] / N_pics_per_worker[1]]
N_bb_votes_required = 2
N_category_votes_required = 2

bb_iou_threshold = 0.5  # how much overlap there must be between bbs

if cv2.__version__ == '3.0.0' or cv2.__version__ == '3.0.0-dev':
    scale_flag = cv2.CASCADE_SCALE_IMAGE
    BGR2GRAYCONST = cv2.COLOR_BGR2GRAY
#    FACECONST = cv2.face
    FACECONST = cv2  # i am not sure what this fixes but it breaks fisherface
    # now i understand, if you havent installed contrib then even if you have cv2 v3 then cv2.face does not exist
    HAARCONST = cv2.CASCADE_SCALE_IMAGE


else:
    scale_flag = cv2.cv.CV_HAAR_SCALE_IMAGE
    BGR2GRAYCONST = cv2.cv.CV_BGR2GRAY
    FACECONST = cv2
    HAARCONST = cv2.cv.CV_HAAR_SCALE_IMAGE


def jp_categories():
    jp = db.products_fp
    cat_dict = paperdoll_shopstyle_women
    jp_dict = {}
    for key, value in cat_dict.iteritems():
        a = db.products_jp.find_one({'categories.id': value})
        if a:
            one_dict = {'id': a['categories'][0]['localizedId'], 'name': a['categories'][0]['name']}
            jp_dict[key] = one_dict
        else:
            jp_dict[key] = {}
    return jp_dict


shopstyle_paperdoll_women = {'bootcut-jeans': 'jeans',
                             'classic-jeans': 'jeans',
                             'cropped-jeans': 'jeans',
                             'distressed-jeans': 'jeans',
                             'flare-jeans': 'jeans',
                             'relaxed-jeans': 'jeans',
                             'skinny-jeans': 'jeans',
                             'straight-leg-jeans': 'jeans',
                             'stretch-jeans': 'jeans',
                             'jeans': 'jeans',
                             'womens-suits': 'suit',
                             'shorts': 'shorts',
                             'blazers': 'blazer',
                             'tees-and-tshirts': 't-shirt',
                             'womens-tops': 'top',
                             'button-front-tops': 'top',
                             'camisole-tops': 'top',
                             'cashmere-tops': 'top',
                             'halter-tops': 'top',
                             'longsleeve-tops': 'top',
                             'shortsleeve-tops': 'shirt',
                             'sleeveless-tops': 'top',
                             'tank-tops': 'top',
                             'tunic-tops': 'blouse',
                             'polo-tops': 'top',
                             'skirts': 'skirt',
                             'mini-skirts': 'skirt',
                             'mid-length-skirts': 'skirt',
                             'long-skirts': 'skirt',
                             'sweaters': 'sweater',
                             'sweatshirts': 'sweatshirt',
                             'cashmere-sweaters': 'sweater',
                             'cardigan-sweaters': 'cardigan',
                             'crewneck-sweaters': 'sweater',
                             'turleneck-sweaters': 'sweater',
                             'v-neck-sweaters': 'sweater',
                             'womens-pants': 'pants',
                             'wide-leg-pants': 'pants',
                             'skinny-pants': 'pants',
                             'dress-pants': 'pants',
                             'cropped-pants': 'pants',
                             'casual-pants': 'pants',
                             'dresses': 'dress',
                             'cocktail-dresses': 'dress',
                             'day-dresses': 'dress',
                             'evening-dresses': 'dress',
                             'jackets': 'jacket',
                             'neckless-jackets': 'jacket',
                             'casual-jackets': 'jacket',
                             'leather-jackets': 'jacket',
                             'vests': 'cardigan',
                             'coats': 'coat',
                             'womens-outerwear': 'coat',
                             'plus-size-outerwear': 'coat',
                             'fur-and-shearling-coats': 'coat',
                             'leather-and-suede-coats': 'coat',
                             'puffer-coats': 'coat',
                             'raincoats-and-trenchcoats': 'coat',
                             'wool-coats': 'coat',
                             'leggings': 'tights'}



'''
this is a list of categories under womens-clothing and mens-clothing in case its needed
TOP LEVEL CLOTHES:
 u'womens-outerwear',
 u'shorts',
 u'jackets',

['womens-clothes',

 u'womens-outerwear',
 u'wool-coats',
 u'coats',
 u'puffer-coats',
 u'fur-and-shearling-coats',
 u'raincoats-and-trenchcoats',
 u'leather-and-suede-coats',

 u'jewelry',
 u'diamond-jewelry',
 u'diamond-necklaces',
 u'diamond-earrings',
 u'diamond-bracelets',
 u'diamond-rings',
 u'charms',
 u'necklaces',
 u'earrings',
 u'bracelets',
 u'brooches-and-pins',
 u'rings',
 u'watches',

 u'shorts',

 u'jackets',
 u'blazers',
 u'neckless-jackets',
 u'casual-jackets',
 u'vests',
 u'leather-jackets',

 u'skirts',
 u'mini-skirts',
 u'mid-length-skirts',
 u'long-skirts',

 u'womens-suits',

 u'jeans',
 u'classic-jeans',
 u'cropped-jeans',
 u'skinny-jeans',
 u'stretch-jeans',
 u'straight-leg-jeans',
 u'distressed-jeans',
 u'flare-jeans',
 u'bootcut-jeans',
 u'relaxed-jeans',

 u'womens-tops',
 u'tees-and-tshirts',
 u'cut-sew-tops',
 u'camisole-tops',
 u'button-front-tops',
 u'tank-tops',
 u'tunic-tops',
 u'sleeveless-tops',
 u'halter-tops',
 u'polo-tops',

 u'sweaters',
 u'v-neck-sweaters',
 u'cardigan-sweaters',
 u'crewneck-sweaters',
 u'turleneck-sweaters',
 u'ensembles',

 u'womens-pants',
 u'dress-pants',
 u'casual-pants',
 u'cropped-pants',
 u'skinny-pants',
 u'leggings',
 u'wide-leg-pants',

 u'sweatshirts',

#NOTE WOMENS ATHLETIC SEEMS TO HAVE GONE BYE-BYE
 u'womens-athletic-clothes',
 u'athletic-shorts',
 u'athletic-jackets',
 u'athletic-skirts',
 u'sports-bras-and-underwear',
 u'athletic-tops',
 u'athletic-pants',

#APPARENTLY THIS CATEGORY IS NO MORE
 u'bridal',
 u'bridesmaid',
 u'bridesmaid-jewelry',
 u'bridal-bridesmaid-dresses',
 u'bridesmaid-bags',
 u'bride',
 u'bridal-lingerie',
 u'bridal-gowns',
 u'bridal-shoes',
 u'bridal-jewelry',
 u'bridal-bags',
 u'bridal-veils',

#GONE WITH THE WIND
 u'maternity-clothes',
 u'maternity-outerwear',
 u'maternity-intimates',
 u'maternity-shorts',
 u'maternity-jackets',
 u'maternity-skirts',
 u'maternity-jeans',
 u'maternity-tops',
 u'maternity-sweaters',
 u'maternity-pants',
 u'maternity-dresses',
 u'maternity-swimsuits',

 u'dresses',
 u'bridal-dresses',
 u'bridesmaid-dresses',
 u'wedding-dresses',
 u'cocktail-dresses',
 u'evening-dresses',
 u'day-dresses',

 u'womens-intimates',
 u'undershirts',
 u'camisoles',
 u'panties',
 u'chemises',
 u'socks',
 u'hosiery',
 u'thongs',
 u'pajamas',
 u'bras',
 u'roomwear',
 u'slippers',
 u'robes',
 u'shapewear',

 u'plus-sizes',
 u'plus-size-outerwear',
 u'plus-size-shorts',
 u'plus-size-jackets',
 u'plus-size-skirts',
 u'plus-size-suits',
 u'plus-size-jeans',
 u'plus-size-tops',
 u'plus-size-sweaters',
 u'plus-size-pants',
 u'plus-size-sweatshirts',
 u'plus-size-dresses',
 u'plus-size-intimates',
 u'plus-size-swimsuits',

 u'petites',
 u'petite-outerwear',
 u'petite-shorts',
 u'petite-jackets',
 u'petite-skirts',
 u'petite-suits',
 u'petite-jeans',
 u'petite-tops',
 u'petite-sweaters',
 u'petite-pants',
 u'petite-sweatshirts',
 u'petite-dresses',
 u'petite-intimates',
 u'petite-swimsuits',

 u'womens-accessories',
 u'womens-eyewear',
 u'sunglasses',
 u'womens-eyeglasses',
 u'key-chains',
 u'gift-cards',
 u'straps',
 u'womens-tech-accessories',
 u'belts',
 u'scarves',
 u'womens-umbrellas',
 u'hats',
 u'gloves',

 u'swimsuits',
 u'one-piece-swimsuits',
 u'two-piece-swimsuits',
 u'swimsuit-coverups']

In [19]: get_all_subcategories(db.categories,'mens-clothes')
Out[19]:
['mens-clothes',
 u'mens-outerwear',
 u'mens-coats',
 u'mens-jackets',
 u'mens-denim-jackets',
 u'mens-overcoats-and-trenchcoats',
 u'mens-leather-and-suede-coats',
 u'mens-shorts',
 u'mens-sweatshirts',
 u'mens-suits',
 u'mens-jeans',
 u'mens-slim-jeans',
 u'mens-straight-leg-jeans',
 u'mens-distressed-jeans',
 u'mens-bootcut-jeans',
 u'mens-relaxed-jeans',
 u'mens-loose-jeans',
 u'mens-low-rise-jeans',
 u'mens-shirts',
 u'mens-tees-and-tshirts',
 u'mens-dress-shirts',
 u'mens-polo-shirts',
 u'mens-shortsleeve-shirts',
 u'mens-longsleeve-shirts',
 u'mens-sleepwear',
 u'mens-slippers',
 u'mens-pajamas',
 u'mens-robes',
 u'mens-sweaters',
 u'mens-v-neck-sweaters',
 u'mens-cardigan-sweaters',
 u'mens-crewneck-sweaters',
 u'mens-turtleneck-sweaters',
 u'mens-half-zip-sweaters',
 u'mens-vests',
 u'mens-ties',
 u'mens-pants',
 u'mens-casual-pants',
 u'mens-cargo-pants',
 u'mens-dress-pants',
 u'mens-chinos-and-khakis',
 u'mens-athletic',
 u'mens-athletic-shorts',
 u'mens-athletic-jackets',
 u'mens-athletic-shirts',
 u'mens-athletic-pants',
 u'mens-underwear-and-socks',
 u'mens-tee-shirts',
 u'mens-socks',
 u'mens-boxers',
 u'mens-briefs',
 u'mens-big-and-tall',
 u'mens-big-and-tall-coats-and-jackets',
 u'mens-big-and-tall-shorts',
 u'mens-big-and-tall-suits',
 u'mens-big-and-tall-jeans',
 u'mens-big-and-tall-shirts',
 u'mens-big-and-tall-sweaters',
 u'mens-big-and-tall-pants',
 u'mens-big-and-tall-underwear-and-socks',
 u'mens-accessories',
 u'mens-eyewear',
 u'mens-sunglasses',
 u'mens-eyeglasses',
 u'mens-tech-accessories',
 u'mens-belts',
 u'mens-umbrellas',
 u'mens-hats',
 u'mens-gloves-and-scarves',
 u'mens-wallets',
 u'mens-watches-and-jewelry',
 u'mens-jewelry',
 u'mens-cuff-links',
 u'mens-watches',
 u'mens-swimsuits']
'''

white_list = ["http://www.bunte.de",
              "http://www.gala.fr",
              "yahoo.com",
              "msn.com",
              "yahoo.co.jp",
              "qq.com",
              "uol.com.br",
              "globo.com",
              "naver.com",
              "onet.pl",
              "espn.go.com",
              "news.yahoo.com",
              "163.com",
              "wp.pl",
              "sina.com.cn",
              "news.google.com",
              "bbc.co.uk",
              "cnn.com",
              "news.yandex.ru",
              "rambler.ru",
              "bbc.com",
              "cnet.com",
              "dailymail.co.uk",
              "milliyet.com.tr",
              "nytimes.com",
              "news.mail.ru",
              "zing.vn",
              "sports.yahoo.com",
              "news.yahoo.co.jp",
              "theguardian.com",
              "buzzfeed.com",
              "interia.pl",
              "indiatimes.com",
              "hurriyet.com.tr",
              "huffingtonpost.com",
              "ifeng.com",
              "t-online.de",
              "foxnews.com",
              "drudgereport.com",
              "sohu.com",
              "about.com",
              "weather.com",
              "rediff.com",
              "iqiyi.com",
              "bp.blogspot.com",
              "livedoor.jp",
              "repubblica.it",
              "wunderground.com",
              "vnexpress.net",
              "forbes.com",
              "lemonde.fr",
              "bloomberg.com",
              "mynet.com",
              "telegraph.co.uk",
              "naver.jp",
              "ukr.net",
              "o2.pl",
              "idnes.cz",
              "24h.com.vn",
              "usatoday.com",
              "ig.com.br",
              "news.163.com",
              "washingtonpost.com",
              "spiegel.de",
              "gazeta.pl",
              "rt.com",
              "gismeteo.ru",
              "elpais.com",
              "marca.com",
              "pixnet.net",
              "accuweather.com"]
