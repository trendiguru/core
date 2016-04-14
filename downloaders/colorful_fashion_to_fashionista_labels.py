__author__ = 'jeremy'
from trendi import constants
i=0
for cat in constants.colorful_fashion_parsing_categories:
    newcat=constants.colorful_fashion_to_fashionista[cat]
    cfrp_index = constants.colorful_fashion_parsing_categories.index(cat)
 #   print('oldcat {} newcat {} '.format(cat,newcat))
    if newcat in constants.fashionista_categories:
        fashionista_index = constants.fashionista_categories.index(newcat)
    #    print('oldcat {} newcat {} cfrpix {} fashionistadx {}'.format(cat,newcat,cfrp_index,fashionista_index))
        print('({},{})'.format(cfrp_index,fashionista_index))
    else:
        print('unhandled category:'+str(cat)+' fashionista index:'+str(cfrp_index))
    i=i+1