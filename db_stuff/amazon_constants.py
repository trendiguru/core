
plus_sizes = ['XL', '1X', '2X', '3X', '4X', 'X', 'XX', 'XXX', 'XXXX', 'XXXXX', 'LARGE', 'PLUS']


amazon_categories = {'Dresses': {'childs': ['Cocktail', 'Casual', 'Wedding Party', 'Prom & Homecoming', 'Club']},
                     # under 'Wedding Party' there are sub sub categories
                     'Tops & Tees': {'childs': ['Blouses & Button-Down Shirts', 'Henleys', 'Knits & Tees', 'Polos',
                                                'Tanks & Camis', 'Tunics', 'Vests']},
                     'Sweaters':{'childs': ['Cardigans', 'Pullovers', 'Shrugs', 'Vests']},
                     'Fashion Hoodies & Sweatshirts': {'childs': []},
                     'Jeans': {'childs': []},
                     'Pants': {'childs': ['Casual', 'Night Out & Special Occasion', 'Dress']},
                     'Skirts': {'childs': ['Casual', 'Night Out & Special Occasion']},
                     'Shorts': {'childs': ['Dress', 'Casual', 'Denim', 'Cargo', 'Flat Front', 'Pleated']},
                     'Leggings': {'childs': []},
                     'Active': {'childs': ['Active Hoodies', 'Active Sweatshirts', 'Track & Active Jackets',
                                           'Active Top & Bottom Sets', 'Active Shirts & Tees', 'Active Pants',
                                           'Active Leggings', 'Active Shorts', 'Active Skirts', 'Active Skorts',
                                           'Active Tracksuits']},
                     'Swimsuits & Cover Ups': {'childs': ['Bikinis', 'Tankinis', 'One-Pieces', 'Cover-Ups',
                                                          'Board Shorts', 'Women', 'Rash Guards']},
                     # under 'Women' there are sub sub categories
                     'Jumpsuits, Rompers & Overalls': {'childs': []},
                     'Coats, Jackets & Vests': {'childs': ['Down & Parkas', 'Wool & Pea Coats', 'Denim Jackets',
                                                           'Quilted Lightweight Jackets', 'Casual Jackets',
                                                           'Leather & Faux Leather', 'Fur & Faux Fur', 'Vests',
                                                           'Active & Performance']},
                     # under 'Down & Parkas' there are sub sub categories
                     # under 'Trench, Rain & Anoraks' there are sub sub categories
                     # under 'Active & Performance' there are sub sub categories
                     'Suiting & Blazers': {'childs': ['Blazers', 'Separates', 'Suit Sets']},
                     # under 'Separates' there are sub sub categories
                     'Shirts': {'childs': ['T-Shirts', 'Tank Tops', 'Polos', 'Henleys', 'Casual Button-Down Shirts',
                                           'Dress Shirts', 'Tuxedo Shirts']},
                     'Jackets & Coats': {'childs': ['Active & Performance', 'Down & Down Alternative', 'Fleece',
                                                    'Leather & Faux Leather', 'Lightweight Jackets',
                                                    'Trench & Rain', 'Vests', 'Wool & Blends', 'Outerwear']},
                     # under 'Active & Performance' there are sub sub categories
                     # under 'Lightweight Jackets' there are sub sub categories
                     'Swim': {'childs': ['Trunks', 'Board Shorts', 'Briefs', 'Men', 'Rash Guards']},
                     # under 'Men' there are sub sub categories

                     'Suits & Sport Coats': {'childs': ['Suits', 'Suit Separates', 'Sport Coats & Blazers',
                                                        'Tuxedos', 'Vests']}
                     # under 'Suit Separates' there are sub sub categories
                     }

blacklist = ['Jewelry', 'Watches', 'Handbags', 'Accessories', 'Lingerie, Sleep & Lounge', 'Socks & Hosiery',
             'Handbags & Wallets', 'Shops', 'Girls', 'Boys', 'Shoes', 'Underwear', 'Baby', 'Sleep & Lounge',
             'Socks', 'Novelty & More', 'Luggage & Travel Gear', 'Uniforms, Work & Safety', 'Costumes & Accessories',
             'Shoe, Jewelry & Watch Accessories', 'Traditional & Cultural Wear', 'Active Underwear', 'Active Socks',
             'Active Supporters', 'Active Base Layers', 'Sports Bras', 'Athletic Socks', 'Athletic Supporters']

log_name = '/home/developer/yonti/amazon_download_stats.log'

colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple', 'magenta', 'cyan', 'grey', 'violet',
          'gold', 'silver', 'khaki', 'turquoise', 'brown']

amazon_categories = ['belt',
                             'bikini',
                             'blazer',
                             'blouse',
                             'cardigan',
                             'coat',
                             'dress',
                             'jacket',
                             'jeans',
                             'leggings',
                             'pants',
                             'roampers',
                             'shirt',
                             'shorts',
                             'skirt',
                             'suit',
                             'sweater',
                             'sweatshirt',
                             't-shirt',
                             'tanktop',
                             'tights',
                             'top',
                             'vest',
                             'swimsuit',
                             'unknown',
                             'stockings']