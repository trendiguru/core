import os
import cv2
import pymongo
from redis import Redis
from rq import Queue

redis_conn = Redis(host=os.getenv("REDIS_HOST", "redis1-redis-1-vm"), port=int(os.getenv("REDIS_PORT", "6379")))

features_per_category = {'dress': ['color', 'sleeve_length', 'length'],
                         'top': ['color', 'sleeve_length'],
                         'shirt': ['color', 'sleeve_length'],
                         'blouse': ['color', 'sleeve_length'],
                         't-shirt': ['color', 'sleeve_length'],
                         'skirt': ['color', 'length'],
                         'other': ['color']}

weights_per_category = {'dress': {'color': 0.8, 'sleeve_length': 0.1, 'length': 0.1},
                        'top': {'color': 0.8, 'sleeve_length': 0.2},
                        'shirt': {'color': 0.8, 'sleeve_length': 0.2},
                        'blouse': {'color': 0.8, 'sleeve_length': 0.2},
                        't-shirt': {'color': 0.8, 'sleeve_length': 0.2},
                        'skirt': {'color': 0.9, 'length': 0.1},
                        'other': {'color': 1}}

products_per_ip_pid = {'default':
                                 {'default': 'amazon_US', 'US': 'amazon_US', 'KR': 'GangnamStyle', 'DE': 'amazon_DE'},
                       'fashionseoul':
                                 {'KR': 'GangnamStyle'},
                       '5767jA8THOn2J0DD':
                                 {'KR': 'GangnamStyle'},
                       'RecruitPilot':
                                 {'default': 'recruit'},
                       'recruit-pilot':
                                 {'default': 'recruit'},
                       '6t50LSJxeNEkQ42p':
                                 {'default': 'recruit'},
                       'xuSiNIs695acaHPE':
                                 {'default': 'amaze'},
                       "Rob's Shelter":
                                 {'default': 'amazon_US'},
                       "robsdemartino@yahoo.it":
                                 {'default': 'amazon_US'},
                       "mz1_ND":
                                 {'default': 'amazon_US', 'US': 'amazon_US'},
                       "6nGzEP7cp5s957P4":
                                 {'default': 'amazon_DE'}
                       }
products_per_site = {'default': 'amazon_US', 'fashionseoul.com': 'GangnamStyle', 'fazz.co': 'amazon_US',
                     'plus-model-mag.com': 'Fat_Beauty', 'recruit-lifestyle.co.jp': 'recruit'}

# fingerprint related consts

fingerprint_length = 696
fingerprint_version = 792015  # DayMonthYear
extras_length = 6
histograms_length = [180, 255, 255]
color_fingerprint_weights = [0.05, 0.5, 0.225, 0.225]
K = 0.5                     # for euclidean distance
min_bb_to_image_area_ratio = 0.95  # if bb takes more than this fraction of image area then use  cv2.GC_INIT_WITH_RECT instead of init with mask

#min images sizes , lower than this gets kicked out by paperdoll
minimum_im_width = 50
minimum_im_height = 50

##############
# rq / worker stuff
##############

pd_worker_command =  'cd /home/jeremy/paperdoll3/paperdoll-v1.0/ && /usr/bin/python /usr/local/bin/rqworker -w trendi.matlab_wrapper.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd &'
pd_worker_command_braini1 =  'cd /home/pd_user/paperdoll  && /usr/bin/python /usr/local/bin/rqworker  -w trendi.matlab_wrapper.tgworker.TgWorker  pd &'
pd_worker_command_braini =  'cd /home/pd_user/paperdoll  && /usr/bin/python /usr/local/bin/rqworker  -w trendi.matlab_wrapper.tgworker.TgWorker  pd &'
string_to_look_for_in_pd_command = 'tgworker'
q1 = Queue('start_pipeline', connection=redis_conn)
q2 = Queue('check_if_relevant', connection=redis_conn)
q3 = Queue('manual_gender', connection=redis_conn)

N_expected_pd_workers_per_server = 15
N_expected_pd_workers_per_server_braini1 = 47
N_default_workers = 47

#general queues on braini
string_to_look_for_in_rq_command = 'rqworker'
unique_strings_to_look_for_in_rq_command = ['person_job','tgworker','item_job','merge_items','merge_people,''start_pipeline','person_job']   #,'fingerprint_new'  ,'find_similar
# the worker_commands are ordered by priority of queue
worker_commands =['/usr/bin/python /usr/local/bin/rqworker person_job &',
                   'cd /home/pd_user/paperdoll  && /usr/bin/python /usr/local/bin/rqworker  -w trendi.matlab_wrapper.tgworker.TgWorker  pd &',
                  '/usr/bin/python /usr/local/bin/rqworker item_job &',
                '/usr/bin/python /usr/local/bin/rqworker merge_items &',
                '/usr/bin/python /usr/local/bin/rqworker start_pipeline &',
                '/usr/bin/python /usr/local/bin/rqworker person_job &'
                  ]
 #                 '/usr/bin/python /usr/local/bin/rqworker fingerprint_new &',,

multi_queue_command ='/usr/bin/python /usr/local/bin/rqworker  start_pipeline person_job item_job merge_items merge_people wait_for_person_ids'
unique_in_multi_queue = 'wait_for_person_ids'
N_expected_workers_by_server={'braini1':45,'brain2':45,'brain3':90,'braini4':90,'braini5':90}
N_max_workers = 120
lower_threshold = 70
upper_threshold = 85
neurodoorman_queuename = 'neurodoor'
neurodooll_queuename = 'neurodoll'
#########
# DB stuff
#########
#for google cloud servers, environment line in /etc/supervisor.conf should be:
#environment=REDIS_HOST="redis1-redis-1-vm",REDIS_PORT=6379, MONGO_HOST="mongodb1-instance-1",MONGO_PORT=27017

#for non- google cloud , environment line in /etc/supervisor.conf should be:
#environment=REDIS_HOST="localhost",REDIS_PORT=6379,MONGO_HOST="localhost",MONGO_PORT=27017

# to do the portforwards required to make this work:
#ssh -f -N -L 27017:mongodb1-instance-1:27017 -L 6379:redis1-redis-1-vm:6379 root@extremeli.trendi.guru
#to kill nound ports
# lsof -ti:27017 | xargs kill -9
# lsof -ti:6379 | xargs kill -9
#to add to .bashrc (maybe better in .profile!!)
#export REDIS_HOST="localhost"
#export REDIS_PORT=6379
#export MONGO_HOST="localhost"
#export MONGO_PORT=27017

parallel_matlab_queuename = 'pd'
nonparallel_matlab_queuename = 'pd_nonparallel'
caffe_path_in_container = '/opt/caffe'
db = pymongo.MongoClient(host=os.getenv("MONGO_HOST", "mongodb1-instance-1"),
                         port=int(os.getenv("MONGO_PORT", "27017"))).mydb

#db = pymongo.MongoClient(host="mongodb1-instance-1").mydb
#redis_conn = Redis(host="redis1-redis-1-vm")
# new worker : rqworker -u redis://redis1-redis-1-vm:6379 [name] &
redis_conn_old = Redis()
update_collection_name = 'products'

############
## caffe stuff
############

caffeRelevantLabels = [601, 608, 610, 614, 617, 638, 639, 655, 689, 697, 735, 775, 834, 841, 264, 401, 400]

# fp rating related constants
nn_img_minimum_sidelength = 10  #for traiing cnn's , min width and height
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

paperdoll_relevant_categories = ['top', 'pants', 'shorts', 'jeans', 'jacket', 'blazer', 'shirt', 'skirt', 'blouse','dress',
                                 'bodysuit', 'vest', 'cardigan', 'leggings', 'sweater', 't-shirt', 'coat',
                                 'suit', 'tights', 'sweatshirt', 'stockings']

ultimate_21_to_paperdoll = [None,None,None,5,16,9,None,None,None,None,3,13,1,None,2,None,7,20,17,14,8]


#used for pixel level output of neurodoll as of 260716
ultimate_21 = ['bgnd','bag','belt','blazer','coat','dress','eyewear','face','hair','hat',
               'jeans','leggings','pants','shoe','shorts','skin','skirt','stockings','suit','sweater',
               'top']

#used for pixel level output of neurodoll as of 260716
ultimate_21_dict = {'bag': 1, 'belt': 2, 'bgnd': 0, 'blazer': 3, 'coat': 4, 'dress': 5, 'eyewear': 6, 'face': 7, 'hair': 8, 'hat': 9,
    'jeans': 10, 'legging': 11, 'pants': 12, 'shoe': 13, 'shorts': 14, 'skin': 15,  'skirt': 16, 'stocking': 17, 'suit': 18, 'sweater': 19, 'top': 20}

web_tool_categories_v1_to_ultimate_21 = [1, 2, 3, 19, 4, 5, 6, 13, 9, 4, 10, 12, 14, 16, 17, 18,19, 20, None, None, None]


tamara_berg_improved_categories = ['background','belt','dress','eyewear','footwear','hat','legging','outerwear','pants','skirts',
                                   'top','skin','BAG????','tights','shorts','blouse','bra','vest','suit','jeans',
                                   'necklace','sweatshirt','tie']

tamara_berg_categories = ['bag', 'belt', 'dress', 'eyewear', 'footwear', 'hat', 'legging', 'outerwear', 'pants','skirts',
                          'top', 'skin', 'background']   # orig t.b. cats don't have skin or bg

#these are in use for multilabeller output as of 260716 - will prob change to v2 in near future so i can use
#results of filipino categorization efforts
#20 cats
web_tool_categories = ['bag', 'belt', 'blazer','cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket',
                       'jeans','pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini',
                       'womens_swimwear_nonbikini']

#removed blazer and added 'overalls','sweatshirt', 'bracelet','necklace','earrings','watch',
#25 cats
web_tool_categories_v2 = ['bag', 'belt', 'cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket','jeans',
                     'pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini','womens_swimwear_nonbikini',
                    'overalls','sweatshirt' , 'bracelet','necklace','earrings','watch' ]

tamara_berg_to_web_tool = [0, 1, 5, 6, 7, 8, 14, 4, 11, 13, 17, None, None]
tamara_berg_to_web_tool_dict = {'bag':'bag','belt':'belt','dress':'dress','eyewear':'eyewear','footwear':'footwear',
                                'hat':'hat','legging':'stocking','outerwear':'coat','pants':'pants','skirts':'skirt','top':'top'}
#missing from webtool compared ot tamara_berg - 'legging', 'outerwear','skin',

paperdoll_paperdoll_men = {'top': 'shirt', 'pants': 'pants', 'shorts': 'shorts', 'jeans': 'jeans', 'jacket': 'jacket',
                           'blazer': 'jacket', 'shirt': 'shirt', 'skirt': 'pants', 'blouse': 'shirt',
                           'dress': 'suit', 'sweater': 'sweater', 't-shirt': 'shirt', 'bodysuit': 'suit',
                           'cardigan': 'sweater', 'coat': 'coat', 'suit': 'suit', 'vest': 'vest',
                           'sweatshirt': 'sweatshirt', 'leggings': 'pants', 'stockings': 'pants', 'tights': 'pants'}

paperdoll_categories = {"whole_body": ['bodysuit', 'dress', 'jumper', 'suit', 'romper'],
                        "upper_cover": ['blazer', 'cape', 'jacket', 'cardigan', 'coat', 'vest', 'sweatshirt'],
                        "upper_under": ['t-shirt', 'blouse', 'shirt', 'top', 'sweater', 'sweatshirt'],
                        "lower_cover": ['shorts', 'skirt', 'jeans', 'pants'],
                        "lower_under": ['stockings', 'tights', 'leggings']}



nn_categories = {"whole_body": ['dress', 'suit'],
                 "upper_cover": ['blazer', 'coat'],
                 "upper_under": ['top', 'sweater'],
                 "lower_cover": ['shorts', 'skirt', 'jeans', 'pants', 'belt'],
                 "lower_under": ['stocking', 'legging']}
                 # "feet_cover": ['shoes', 'boots', 'loafers', 'flats', 'sneakers', 'clogs', 'heels', 'wedges',
                 #                'pumps', 'sandals'],
                 # "feet_under": ['socks']}

fash_augmented_that_didnt_get_into_nn_categories = ['bag','purse','scarf','hat','accessories','sunglasses','glasses',
                                    'intimate','necklace','bracelet','ring','earrings','gloves','watch',
                                    'wallet','hair','skin','face'] #

#for our purposes -
# blazer is a suit jacket (without the pants)
# coat is a winter coat
# jacket is a winter jacket (not a suit jacket)
# suit has a jacket and pants
binary_cats = ['bag', 'belt','bikini','blazer', 'bracelet', 'cardigan', 'coat', 'dress', 'earrings',
               'eyewear','footwear', 'gloves','hat', 'jacket', 'jeans', 'lingerie',  'necklace',
               'overalls','pants', 'ring', 'scarf', 'shorts', 'skirt', 'stocking', 'suit', 'sweater',
               'sweatshirt','swimwear%20NOT%20bikini%AND%20woman', 'swimwear%20AND%20man','top', 'watch']

exclude_terms_for_binary_cats = [None,['conveyor','boxing','heavyweight']

missing_from_v2_compared_to_binary_cats = [ 'blazer',  'gloves', 'lingerie', 'ring', 'swimwear%20AND%20man']

#web_tool_v2=['bag', 'belt',       'cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket',
#                         'jeans','pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini',
#                       'womens_swimwear_nonbikini','overalls','sweatshirt', 'bracelet','necklace','earrings','watch' ]




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

# these are the fashionista db cats in order , e.g. the mask will have 1 for null (bkgnd) and 56 for skin
#the first '' value is to keep mask=1 -> null, mask=2->tights etc
fashionista_categories = ['','null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt',
                          'purse','boots','blouse','jacket','bra','dress','pants','sweater','shirt','jeans',
                          'leggings','scarf','hat','top','cardigan','accessories','vest','sunglasses','belt','socks',
                          'glasses','intimate','stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels',
                          'wedges','ring','flats','tie','romper','sandals','earrings','gloves','sneakers','clogs',
                          'watch','pumps','wallet','bodysuit','loafers','hair','skin']

fashionista_categories_augmented = ['','null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse',
                                    'boots','blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings',
                                    'scarf','hat','top','cardigan','accessories','vest','sunglasses','belt','socks','glasses',
                                    'intimate','stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges',
                                    'ring','flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch',
                                    'pumps','wallet','bodysuit','loafers','hair','skin','face']  #0='',1='null', 57='face'

fashionista_categories_augmented_zero_based = ['null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse',
                                    'boots','blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings',
                                    'scarf','hat','top','cardigan','accessories','vest','sunglasses','belt','socks','glasses',
                                    'intimate','stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges',
                                    'ring','flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch',
                                    'pumps','wallet','bodysuit','loafers','hair','skin','face']  #0='bk', 56='face'

fashionista_categories_for_conclusions = {'background':0,'tights':1,'shorts':2,'blazer':3,'t-shirt':4,'bag':5,'shoes':6,'coat':7,'skirt':8,'purse':9,
                                    'boots':10,'blouse':11,'jacket':12,'bra':13,'dress':14,'pants':15,'sweater':16,'shirt':17,'jeans':18,'leggings':19,
                                    'scarf':20,'hat':21,'top':22,'cardigan':23,'accessories':24,'vest':25,'sunglasses':26,'belt':27,'socks':28,'glasses':29,
                                    'intimate':30,'stockings':31,'necklace':32,'cape':33,'jumper':34,'sweatshirt':35,'suit':36,'bracelet':37,'heels':38,'wedges':39,
                                    'ring':40,'flats':41,'tie':42,'romper':43,'sandals':44,'earrings':45,'gloves':46,'sneakers':47,'clogs':48,'watch':49,
                                    'pumps':50,'wallet':51,'bodysuit':52,'loafers':53,'hair':54,'skin':55,'face':56}


colorful_fashion_parsing_categories = ['bk','T-shirt','bag','belt','blazer','blouse','coat','dress','face','hair','hat',
'jeans','legging','pants','scarf','shoe','shorts','skin','skirt','socks','stocking','sunglass','sweater']

colorful_fashion_to_fashionista = {'bk':'null','T-shirt':'t-shirt','bag':'bag','belt':'belt','blazer':'blazer','blouse':'blouse',
            'coat':'coat','dress':'dress','face':None,'hair':'hair','hat':'hat','jeans':'jeans','legging':'leggings',
            'pants':'pants','scarf':'scarf','shoe':'shoes','shorts':'shorts','skin':'skin','skirt':'skirt','socks':'socks',
            'stocking':'stockings','sunglass':'sunglasses','sweater':'sweater'}

#all the cf stuff maps directly to fashionista except skin (index 8) which I map to 56 (end of fashionista list)
#currently off by one
colorful_fashion_to_fashionista_index_conversion = [(0,0),(1,4),(2,5),(3,27),(4,3),(5,11),(6,7),(7,14),(8,56),
        (9,54),(10,21),(11,18),(12,19),(13,15),(14,20),(15,6),(16,2),(17,55),(18,8),(19,28),(20,31),(21,26),(22,16)]

#I think nadav used 'baground' ie bag actually means background, so maybe background (label 12) means bag

#21 cats for direct replacement of VOC systems
#lose the necklace,
#combine tights and leggings


tamara_berg_to_ultimate_21_index_conversion = [(0,1),(1,2),(2,5),(3,6),(4,13),(5,9),(6,11),(7,4),(8,12),(9,16),(10,20)]

tamara_berg_improved_to_ultimate_21_index_conversion = [(0,1),(1,2),(2,5),(3,6),(4,7),(5,8),(6,10),(7,11),(8,12),(9,15),
                  (10,19),(11,14),(12,0),(13,10),(14,13),(15,19),(16,4),(17,20),(18,16),(19,9),
                  (20,0),(21,19),(22,18),(23,19)]

fashionista_to_ultimate_21_index_conversion = []


pascal_context_labels = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','table','dog','horse','motorbike','person','pottedplant',\
'sheep','sofa','train','tvmonitor','bag','bed','bench','book','building','cabinet','ceiling','cloth','computer','cup','door','fence',\
'floor','flower','food','grass','ground','keyboard','light','mountain','mouse','curtain','platform','sign','plate','road','rock',\
'shelves','sidewalk','sky','snow','bedclothes','track','tree','truck','wall','water','window','wood']

#In a universe of outerwear, dress, pants, skirt, top, what can't be worn together (dont bug me about shorts under skirt)
#and yes this contains redundant information,  if a cant be worn with b then opp is true
exclusion_relations = {'dress':['skirt','pants','top'],'pants':['dress','skirt'], 'skirt':['dress','pants'],'top':['dress'],  'outerwear':[]}

#ebay shop the look
#ebay clothing categories
#http://wwd.com/fashion-dictionary/#
#http://www.modcloth.com/style_glossary
#see e.g. http://s1073.photobucket.com/user/Sofialiciel/media/Fashion%20Dictionary/fashion-ill-10.jpg.html?sort=3&o=9
taxonomy_leaves = ['mini_dress','midi_dress','maxi_dress','cocktail_dress','strapless_dress','sari_dress','pencil_dress','bustle_dress',
                   'empire_dress','tunic_dress','dropped-waist_dress','shirtwaist_dress','sheath_dress','A-line_dress','tent_dress',

                    'mini_skirt','midi_skirt','maxi_skirt','bustle_skirt','bubble_skirt','broomstick_skirt','pencil_skirt','cowl-drape_skirt','sarong-wrap_skirt','tulip_skirt'
                   'full_skirt','A-line_skirt','fitted_skirt','dirndl_skirt','trumpet_skirt','wrap_skirt','gore_skirt','godet_skirt','tiered_skirt','handkerchief_skirt','layered_skirt',

                    'classic_jeans','cropped_jeans','distressed_jeans','flare_jeans','relaxed-jeans','skinny_jeans','straight-leg_jeans',
                    'stretch_jeans','bootcut_jeans','classic_jeans',

                    'dress_pants','trouser_pants','harem_pants','overall_pants','pegged_pants','dhoti_pants','jodhpur_pants','sailor_pants','bell-bottom_pants','sweatpants',

                    'oxford_top','blouse_top','sweat-shirt_top', #longsleeve
                    'tank_top','spaghetti-strap_top','halter_top', 'tube_top','bandeau_top' #sleeveless
                    't-shirt_top','polo_shirt','henley_shirt',  #short sleeve
                    'bustier_top','vest_top','camp_top','cossack_top',
                   'tuxedo_top','artists-smock_top','surplice-wrap_top','gypsy_top','shell_top','sailor_top',

                    'pump_shoe','tennis_shoe','sandal_shoe','flats_shoe','boots_shoe','hiking_shoe',
                    'thigh-high_shoe','knee-high_shoe','wellington_boot_shoe','cowboy_boot_shoe','timberland_boot_shoe',
                    'wedge_shoe','loafer_shoe','converse_shoe','oxford_shoe','docksider_shoe','pump_shoe','high-heel_shoe',

                    'trunks_swimwear','one-piece_swimwear','bikini_swimwear',

                    'hat_accessory','belt_accessory','scarf_accessory','tie_accessory','necklace_accessory','purse_accessory',

                    'evening_glove','mitten_glove','gauntlet_glove','short_glove','wrist-length_glove',

                    'winter_coat','trench_coat','cape_coat','tent_coat','cocoon_coat','coachman_coat','stadium_coat','trench_coat',
                   'redingote_coat','wrap_coat','chesterfield_coat','parka_coat','duffel_coat','polo_coat','chesterfield_coat',

                    'suit_jacket','skiing_jacket','blazer_jacket','cardigan_jacket','smoking_jacket','bellboy_jacket','eisenhower_jacket',
                   'bolero_jacket','norfolk_jacket','pea_jacket','safari_jacket','motorcycle_jacket','baseball_jacket','cricket_jacket','regency_jacket',

                    'cardigan_sweater','sweater',

                   'hats',

                   'shoes',

                   'windsor_collar','spread_collar','point_collar','rounded_collar','open_tab_collar','closed_tab_collar',
                   'pin_collar','button-down_collar',

                   'peaked_lapel','peaked_shawl_lapel','cloverleaf_lapel','rounded-shoulder_lapel','squared-shoulder_lapel',
                   'dropped_shoulder_lapel','built-up-shoulder_lapel','fish-mouth_lapel','t-shaped_lapel',

                   'french_cuff','rollback_cuff','shirt_tailored_cuff','wrapped_cuff','button_loops_cuff','split_cuff','wrapped_cuff','button-tab_cuff',
                   'gauntlet_cuff','zipper_cuff','ribcasing_cuff','adjustable-tab_cuff','belted_cuff','buccaneer_cuff','western-snapped_cuff',

                   'elastic_sewing','smocking_sewing','drawstring_sewing','lacing_sewing',

                   'waistlines',

                   'pockets',


                   'handbags',

                   'edges',

                   'pleats',
                   'collars',
                   'sewing_treatments','elastic_band_treatment','smocking_treatment','drawstring_treatment','lacing_treatment'     ,
                    ]

#from shopstyle categories
mega_categories = ['handbag',
                   'bridal',
                   'jeans',
                   'womens-intimates',
                   'womens-athletic-clothes',
                    'bridal',
                    'jeans',
                    'dresses',
          'womens-intimates',
          'jackets',
          'jewelry',
          'maternity-clothes',
          'womens-outerwear',
          'womens-pants',
          'petites',
          'plus-sizes',
          'shorts',
          'skirts',
          'womens-suits',
          'sweaters',
          'swimsuits',
          'sweatshirts',
          'teen-girls-clothes',
          'womens-tops',
          'gloves',
          'hats',
          'scarves',
          'womens-eyewear',
          'sunglasses',
         'athletic-shorts',
          'athletic-skirts'
          'sports-bras',  #from sports-bras-and-underwear
          'athletic-jackets',
          'athletic-pants',
          'athletic shorts',
          'athletic-tops',
          'lingerie',
          'shoes',
          'bridal-gowns',
          'bridal-veils',
          'bridesmaid-dresses',
          'bridesmaid-bags',
          'classic-jeans',
          'cropped-jeans',
          'distressed-jeans',
          'flare-jeans',
          'relaxed-jeans',
          'skinny-jeans',
          'straight-leg-jeans',
          'stretch-jeans',
          'bootcut-jeans'
          'classic-jeans'
          'cocktail-dresses'
          'day-dresses',
          'evening-dresses',
          'womens-intimates',
          'bras',
          'camisoles'
          'chemises',
          'gowns',
          'nightgowns',
          'hosiery',
          'pajamas',
          'panties',
          'robes',
          'socks',
          'shapewear',
          'slippers',
          'thongs',
          'jackets',
          'blazers',
          'casual-jackets',
          'vests',
          'leather-jackets',
          'maternity-dresses',
          'maternity-pants',
          'maternity-shorts',
          'maternity-skirts',
          'maternity-swimsuits',
          'coats']

#  u'maternity-tops'
#  u'maternity-sweaters',
#  u'maternity-outerwear',
#  u'maternity-intimates',
#  u'maternity-jackets',

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
paperdoll_ttl = 300  # seconds to wait for paperdoll result
caffe_general_ttl = 30  # seconds to wait for paperdoll result
general_ttl = 2000  # ttl of all queues

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

if cv2.__version__ == '3.0.0' or cv2.__version__ == '3.0.0-dev' or cv2.__version__ == '3.1.0':
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

#if any one of these appears as a substring in a url, it gets booted
blacklisted_terms = {'alohatube',
    'asshole',
     'asswipe',
    'beeg.com',
    'bing.com'
     'b00b',
     'boob',
     'bitch',
     'blowjob',
     'boffing',
     'butt-pirate',
     'c0ck',
     'clit',
     'cock',
     'cock-goblins',
     'cunt',
     'dominatrix',
     'ejaculate',
     'enema',
     'faggot',
     'fuck',
    'google.com',
    'google.co.il',
    'google.it',
    'google.fr',
     'jackoff',
     'jerkoff',
    'jizz',
   'jjgirls',
     'masturb',
     'muffplower',
     'mrsex',
    'mygrandmatube',
  'naughtyamerica'
     'nutsack',
     'orgasm',
     'p0rn',
     'penis',
    'pichunter',
     'porn',
     'pr0n',
     'pussy',
     'redtube',
     'schlampe',
     'schlong',
     'semen',
     'sex.com',
    'sexychinese',
     'streamsex',
    'shit',
     'skank',
     'slut',
     'smut',
     'testicle',
     'tits',
     'twat',
    'valenciacitas.com'
     'wank',
     'wh00r',
     'wh0re',
     'whore',
     'wikipedia.com',
     'x-rated',
    'xhamster'
     'xnxx',
     'xrated',
     'xxx',
    'youjizz'}

#....unless any of these also appears
blacklisted_exceptions = {'xxxlarge',
                          'xxxsmall',
                          'peacock',
                          'cocktail',
                          'blogspot',
                          'wellandgood',
                          'fashion',
                          'xxxhairsnap',
                          'kannada'}
