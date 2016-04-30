__author__ = 'jeremy'
from trendi import constants

def color_fashion_to_fashionista():
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

def tamara_berg_improved_to_ultimate_21():
    i=0
    for ind in range(0,len(constants.tamara_berg_improved_categories)):
        tup = constants.tamara_berg_improved_to_ultimate_21_index_conversion[ind]
        ultimate21_index=tup[1]
        if ultimate21_index <0:
            print('unhandled category:'+str(ind)+' berg label:'+str(constants.tamara_berg_improved_categories[ind]))
        else:
            print('oldcat {} {} newcat {} {} '.format(ind,constants.tamara_berg_improved_categories[ind],ultimate21_index,constants.ultimate_21[ultimate21_index]))

if __name__=="__main__":
    tamara_berg_improved_to_ultimate_21()