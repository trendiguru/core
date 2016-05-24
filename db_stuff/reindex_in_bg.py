from .. import constants

db = constants.db


def getIndexesNames(coll):
    idx_info  = coll.index_information()
    keys = idx_info.keys()
    keys.remove('_id_')
    #removes the '_1' from the key names
    keys = [k[:-2] for k in keys]




def reIndex(collection_name):
    collection = db[collection_name]
    oldIndexes = getIndexesNames(collection)
    #remove indexes
    collection.drop_indexes()
    #build new indexes
    for index in oldIndexes:
        collection.create_index(index, background=True)
    print('Index done!')