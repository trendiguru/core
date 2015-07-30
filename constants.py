# file containing constants for general TG use

# fingerprint related consts

fingerprint_length = 56
fingerprint_version = 1
extras_length = 6
histograms_length = 25
K = 0.5                     # for euclidean distance
min_bb_to_image_area_ratio = 0.95  # if bb takes more than this fraction of image area then use cv2.GC_INIT_WITH_RECT instead of init with mask

# fp rating related constants
min_image_area = 400
min_images_per_doc = 10  # item has to have at least this number of pics
max_images_per_doc = 18  # item has to have less than this number of pics
max_items = 50  # max number of items to consider for rating fingerprint

classifiers_folder = "/home/ubuntu/Dev/trendi_guru_modules/classifiers/"

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

# clothes parsing items' legends

RELEVANT_ITEMS = {'2': 'leggings', '3': 'shorts', '4': 'blazers', '5': 'tees-and-tshirts',
                  '8': 'womens-outerwear', '9': 'skirts', '12': 'womens-tops', '13': 'jackets', '14': 'bras',
                  '15': 'dresses', '16': 'womens-pants', '17': 'sweaters', '18': 'womens-tops', '19': 'jeans',
                  '20': 'leggings', '23': 'womens-top', '24': 'cardigan-sweaters', '25': 'womens-accessories',
                  '26': 'mens-vests', '29': 'socks', '31': 'womens-intimates', '32': 'stockings',
                  '35': 'cashmere-sweaters', '36': 'sweatshirts', '37': 'womens-suits', '43': 'mens-ties'}
IRELEVANT_ITEMS = {'1': 'background', '6': 'bag', '7': 'shoes', '10': 'purse', '11': 'boots', '21': 'scarf',
                   '22': 'hats', '27': 'sunglasses', '28': 'belts', '30': 'glasses', '33': 'necklace', '34': 'cape',
                   '38': 'bracelet', '39': 'heels', '40': 'wedges', '41': 'rings',
                   '42': 'flats', '44': 'romper', '45': 'sandals', '46': 'earrings', '47': 'gloves',
                   '48': 'sneakers', '49': 'clogs', '50': 'watchs', '51': 'pumps', '52': 'wallets', '53': 'bodysuit',
                   '54': 'loafers', '55': 'hair', '56': 'skin'}

# for web bounding box interface
# this is for going to the previous item, highest numbered image
max_image_val = 666
svg_folder = '/var/www/static/svgs/'

svg_url_prefix = 'http://extremeli.trendi.guru/static/svgs/'

nadav = 'awesome'

Reserve_cpus = 2  # number of cpus to not use when doing stuff in parallel


# QC worker voting params

# N_top_results - a list of N1, N2 etc where N1 items are shown to first wave of voters,
# then top N2 of those are shown to second wave of voters, etc
# the length of this list is the number of voting stages
N_top_results_to_show = [100, 20]
N_pics_per_worker = [5, 20]
N_workers = [20, 2]
N_bb_votes_required = 2
N_category_votes_required = 2

bb_iou_threshold = 0.5  # how much overlap there must be between bbs
