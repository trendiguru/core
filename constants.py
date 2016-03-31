import os

import cv2
import pymongo
from redis import Redis
from rq import Queue

redis_conn = Redis(host=os.getenv("REDIS_HOST", "redis1-redis-1-vm"), port=int(os.getenv("REDIS_PORT", "6379")))

# getting redis /mongo going: do the tunnels
# ssh -f -N -L 6379:redis1-redis-1-vm:6379 root@extremeli.trendi.guru
# ssh -f -N -L 27017:mongodb1-instance-1:27017 root@extremeli.trendi.guru
# and export the environment variables (better yet put the export in .bashrc)
# export REDIS_HOST=localhost
# export REDIS_PORT=6379
# export MONGO_PORT=27017
# export MONGO_HOST=localhost

manual_gender_domains = ['fashionseoul.com']
products_per_site = {'fashionseoul.com': 'GangnamStyle'}
# file containing constants for general TG use
# fingerprint related consts

fingerprint_length = 696
fingerprint_version = 792015  # DayMonthYear
extras_length = 6
histograms_length = [180, 255, 255]
fingerprint_weights = [0.05, 0.5, 0.225, 0.225]
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
string_to_look_for_in_pd_command = 'tgworker'
q1 = Queue('start_pipeline', connection=redis_conn)
q2 = Queue('person_job', connection=redis_conn)
q3 = Queue('item_job', connection=redis_conn)
q4 = Queue('merge_items', connection=redis_conn)
q5 = Queue('merge_people', connection=redis_conn)
q6 = Queue('wait_for_person_ids', connection=redis_conn)

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

#########
# DB stuff
#########
#for google cloud servers, environment line in /etc/supervisor.conf should be:
#environment=REDIS_HOST="redis1-redis-1-vm",REDIS_PORT=6379, MONGO_HOST="mongodb1-instance-1",MONGO_PORT=27017

#for non- google cloud , environment line in /etc/supervisor.conf should be:
#environment=REDIS_HOST="localhost",REDIS_PORT=6379,MONGO_HOST="localhost",MONGO_PORT=27017

# to do the portforwards required to make this work:
#ssh -f -N -L 27017:mongodb1-instance-1:27017 root@extremeli.trendi.guru
#ssh -f -N -L 6379:redis1-redis-1-vm:6379 root@extremeli.trendi.guru
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

paperdoll_relevant_categories = ['top', 'pants', 'shorts', 'jeans', 'jacket', 'blazer', 'shirt', 'skirt', 'blouse',
                                 'dress', 'bodysuit', 'vest', 'cardigan', 'leggings', 'sweater', 't-shirt', 'coat',
                                 'suit', 'tights', 'sweatshirt', 'stockings']

paperdoll_paperdoll_men = {'top': 'shirt', 'pants': 'pants', 'shorts': 'shorts', 'jeans': 'jeans', 'jacket': 'jacket',
                           'blazer': 'blazer', 'shirt': 'shirt', 'skirt': 'shorts', 'blouse': 'shirt',
                           'dress': 'suit', 'sweater': 'sweater', 't-shirt': 'shirt', 'bodysuit': 'suit',
                           'cardigan': 'cardigan', 'coat': 'coat', 'suit': 'suit', 'vest': 'vest',
                           'sweatshirt': 'sweatshirt', 'leggings': 'pants', 'stockings': 'pants', 'tights': 'pants'}

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
