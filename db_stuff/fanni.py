
import annoy
from ..constants import db

def plantAnnoyForest(col_name, category, num_of_trees, distance_function='angular'):
    """"
    create forest for collection
    """
    forest = annoy.AnnoyIndex(696, distance_function)

    items = db[col_name].find({'categories':category})
    for x,item in enumerate(items):
        v= item['fingerprint']
        forest.add_item(x,v)
        """
        annoy index the items in the order the are inserted to the tree
        when searching the forest - the item index is returned back
        thts why we need to match between items in the database and their annoy index
        """
        db[col_name].update_one({'_id':item['_id']},{'$set':{"AnnoyIndex":x}})

    forest.build(num_of_trees)
    """
    for now the tree is saved only on the database server
    >>> the search can only run on database!!!
    """
    name = '/home/developer/annoyJungle/' + col_name+"/"+category+'_forest.ann'
    forest.save(name)
    print ("%s forest in planted! come here for picnics...")


def plantForests4AllCategories(col_name):
    if 'ShopStyle'in col_name or 'GangnamStyle' in col_name:
        from ..db_stuff import shopstyle_constants
        if 'Male' in col_name:
            categories = list(set(shopstyle_constants.shopstyle_paperdoll_male.values()))
        else:
            categories = list(set(shopstyle_constants.shopstyle_paperdoll_female.values()))
    elif 'ebay' in col_name:
        if 'Male' in col_name or 'Unisex' in col_name:
            from ..db_stuff import shopstyle_constants
            categories = list(set(shopstyle_constants.shopstyle_paperdoll_male.values()))
        else:
            from ..db_stuff import ebay_constants
            categories = list(set(ebay_constants.ebay_paperdoll_women.values()))
    else:
        print('ERROR - Bad collection name')
        return

    for cat in categories:
        plantAnnoyForest(col_name,cat,250)

def plantTheFuckingAmazon():
    '''
    create forests for all the categories in all the collections
    '''
    for collection_main in ["ebay", "ShopStyle", "GangnamStyle"]:
        for gender in ["Male", "Female"]:
            collection_name = collection_main +'_'+gender
            plantForests4AllCategories(collection_name)

    plantForests4AllCategories('ebay_Unisex')
    print ("all forests are ready")


def lumberjack(col_name,category,fingerprint, distance_function='angular', num_of_results=1000):
    """
    use annoy to quickly chop down the database and return only the top 1000 trees
    """
    print('searching for top 1000 items in %s' %(col_name))
    forest = annoy.AnnoyIndex(696, distance_function)
    name = '/home/developer/annoyJungle/' + col_name + "/" + category + '_forest.ann'
    forest.load(name)
    result = forest.get_nns_by_vector(fingerprint,num_of_results)
    print("got it!")
    return result