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

#########
#DB stuff
#########

parallel_matlab_queuename = 'pd'
nonparallel_matlab_queuename = 'pd_nonparallel'
# db = pymongo.MongoClient(host=os.environ["MONGO_HOST"], port=int(os.environ["MONGO_PORT"])).mydb
# redis_conn = Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
db = pymongo.MongoClient(host="mongodb1-instance-1").mydb
redis_conn = Redis(host="redis1-redis-1-vm")
redis_conn_old = Redis()
update_collection_name = 'products'
download_version = 792015  # DayMonthYear


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

db_relevant_items = ['women', 'womens-clothes', 'womens-suits', 'shorts', 'petites', 'blazers', 'tees-and-tshirts',
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

papersoll_whole_body = ['bodysuit', 'dress', 'jumper', 'suit', 'romper', 'intimate']
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

# for web bounding box interface
# this is for going to the previous item, highest numbered image
max_image_val = 666
svg_folder = '/var/www/static/svgs/'

svg_url_prefix = 'http://extremeli.trendi.guru/static/svgs/'

nadav = 'awesome'

Reserve_cpus = 2  # number of cpus to not use when doing stuff in parallel

# for gender id
gender_ttl = 5  # 10 seconds ttl , answer should be nearly immediate
paperdoll_ttl = 9045  # seconds to wait for paperdoll result

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