__author__ = 'jeremy'
from trendi.classifier_stuff.caffe_nns import create_nn_imagelsts
from trendi import constants

def create_multilabel_imagefile(in_docker=True):
    '''
    original cats in db are from this list
    web_tool_categories_v2 = ['bag', 'belt', 'cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket','jeans',
                     'pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini','womens_swimwear_nonbikini',
                    'overalls','sweatshirt' , 'bracelet','necklace','earrings','watch' ]

    destination cats
    hydra_ml_test_cats=['dress','skirt','pants_jeans','footwear']

   @
    :return:
    '''
    catfile = 'hydra_test_ml.txt'
    cat_index=0
    for cat in constants.hydra_ml_test_cats:
        lookfor_this = cat
        if cat=='pants_jeans':  #do both pants, eans as index 2
            create_nn_imagelsts.one_class_positives_from_multilabel_db(image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',
                                           catsfile_dir = '/data/jeremy/image_dbs/labels/hydra_test',catsfile=catfile,
                                           desired_cat='pants',desired_index=cat_index,in_docker=in_docker)
            create_nn_imagelsts.one_class_positives_from_multilabel_db(image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',
                                           catsfile_dir = '/data/jeremy/image_dbs/labels/hydra_test',catsfile=catfile,
                                           desired_cat='jeans',desired_index=cat_index,in_docker=in_docker)

        else:
            create_nn_imagelsts.one_class_positives_from_multilabel_db(image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',
                                           catsfile_dir = '/data/jeremy/image_dbs/labels/hydra_test',catsfile=catfile,
                                           desired_cat=cat,desired_index=cat_index,in_docker=in_docker)

        cat_index=cat_index+1


def write_cats_from_db_to_textfile(image_dir='/data/jeremy/image_dbs/tamara_berg/images',catsfile='/data/jeremy/image_dbs/labels/hydra_test/hydrapaper_ml.txt'):
    '''

    original cats in db are from this list
    web_tool_categories_v2 = ['bag', 'belt', 'cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket','jeans',
                     'pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini','womens_swimwear_nonbikini',
                    'overalls','sweatshirt' , 'bracelet','necklace','earrings','watch' ]

    destination cats
    hydra_ml_test_cats=['dress','skirt','pants_jeans','footwear']

   @
    :return:

    '''
    relevant_list=['dress','skirt','pants','jeans','footwear']
    relevant_indices=[0,1,2,2,3]
    db = constants.db
    cursor = db.training_images.find()
    n_done = cursor.count()
    print(str(n_done)+' docs in db')
    lines_written = 0
    n_consistent = 0
    n_inconsistent = 0
    min_votes_for_positive=2
    max_votes_for_negative=0
    with open(catsfile,'w') as fp:
        for i in range(n_done):
            document = cursor.next()
            url = document['url']
            filename = os.path.basename(url)
            full_path = os.path.join(image_dir,filename)
            items_list = document['items'] #
            print items_list
            hotlist = np.zeros(len(constants.web_tool_categories_v2))
            if not 'already_seen_image_level' in document:
                print('no votes for this doc')
                continue
            if document['already_seen_image_level'] < 2:
                print('not enough votes for this doc')
                continue
            for item in items_list:
                cat = item['category']
                if cat in relevant_list:
                    index = relevant_indices[relevant_list.index(cat)]
                    hotlist[index] = hotlist[index]+1
                else:
                    print('irrelevant category : '+str(cat))

            consistent=all([(elem>=min_votes_for_positive or elem<=max_votes_for_negative) for elem in hotlist])
            n_consistent = n_consistent + consistent
            n_inconsistent = n_inconsistent + int(not(consistent))
            print('consistent:'+str(consistent)+' n_con:'+str(n_consistent)+' incon:'+str(n_inconsistent))
            print('hotlist:'+str(hotlist))
            if(consistent):
                line = str(full_path) +' '+ ' '.join(str(int(n)) for n in hotlist)
                lines_written +=1
                fp.write(line+'\n')
    print(str(lines_written)+' lines written to '+catsfile)
