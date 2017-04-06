__author__ = 'jeremy'
from trendi.classifier_stuff.caffe_nns import create_nn_imagelsts
from trendi import constants

def create_multilabel_imagefile():
    '''
    original cats in db are from this list
    web_tool_categories_v2 = ['bag', 'belt', 'cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket','jeans',
                     'pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini','womens_swimwear_nonbikini',
                    'overalls','sweatshirt' , 'bracelet','necklace','earrings','watch' ]

    destination cats
    hydra_ml_test_cats=['dress','skirt','pants_jeans','footwear']


    :return:
    '''
    catfile = 'hydra_test_ml.txt'
    cat_index=0
    for cat in constants.hydra_ml_test_cats:
        lookfor_this = cat
        if cat=='pants_jeans':  #do both pants, eans as index 2
            create_nn_imagelsts.one_class_positives_from_multilabel_db(image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',
                                           catsfile_dir = '/data/jeremy/image_dbs/labels/hydra_test',catsfile=catfile,
                                           desired_cat='pants',desired_index=cat_index)
            create_nn_imagelsts.one_class_positives_from_multilabel_db(image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',
                                           catsfile_dir = '/data/jeremy/image_dbs/labels/hydra_test',catsfile=catfile,
                                           desired_cat='jeans',desired_index=cat_index)

        else:
            create_nn_imagelsts.one_class_positives_from_multilabel_db(image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',
                                           catsfile_dir = '/data/jeremy/image_dbs/labels/hydra_test',catsfile=catfile,
                                           desired_cat=cat,desired_index=cat_index)

        cat_index=cat_index+1


