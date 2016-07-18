from ..constants import db


def delete_or_and_index(collection_name, index_list, delete_flag=False):
    does_exists = verify_collection_exists(collection_name)
    if does_exists:
        print('%s collection already exists' % collection_name)
    else:
        print('%s collection was created' % collection_name)

    collection = db[collection_name]
    if delete_flag:
        collection.delete_many({})
    indexes = collection.index_information().keys()
    for idx in index_list:
        idx_1 = idx + '_1'
        if idx_1 not in indexes:
            collection.create_index(idx, background=True)


def verify_collection_exists(collection_name):
    collection_names = db.collection_names()
    if collection_name not in collection_names:
        collection = db[collection_name]
        tmp = {}
        collection.insert_one(tmp)
        collection.delete_one({})
        return False
    else:
        return True

