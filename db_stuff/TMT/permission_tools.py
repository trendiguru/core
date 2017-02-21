from ...constants import db


def raw_input_plus(msg):
    tmp = raw_input(msg)
    if len(tmp)<2:
        return False, ''
    return True, tmp


def add_collection_to_user(username):
    current_cols = get_current_cols(username)

    col_list = []
    print ('col adding loop - leave blank to break')
    continue_flag, col = raw_input_plus('enter collection name to add: ')

    while continue_flag:
        col_list.append(col)
        continue_flag, col = raw_input_plus('enter additional collection name: ')

    new_cols = current_cols[:]
    for collection in col_list:
        if collection not in new_cols:
            new_cols.append(collection)

    if len(current_cols) != len(new_cols):
        db.trendi_mongo_users.update_one({'name': username}, {'$set': {'collections': new_cols}})


def get_current_cols(username):
    current_cols = db.trendi_mongo_users.find_one({'name': username})['collections']
    print ('current user cols:')
    for x, col in enumerate(current_cols):
        print ('{}. {}'.format(x, col))
    return current_cols


def remove_collection_from_user(username):
    current_cols = get_current_cols(username)

    print ('which collections to delete? (for ex. 1,2,5)')
    cols = raw_input('')
    cols_to_remove = cols.split(',')
    new_cols = [col for x, col in enumerate(current_cols) if str(x) not in cols_to_remove]

    if len(current_cols) != len(new_cols):
        db.trendi_mongo_users.update_one({'name': username}, {'$set': {'collections': new_cols}})


def divide_collection_to_small_collections(col_name, new_size, username):
    collection = db[col_name]
    count = collection.count()

    new_collections_count = count/new_size + 1

    current_cols = get_current_cols(username)

    for i in xrange(new_collections_count):
        items_cursor = collection.find().skip(i*new_size).limit(new_size)
        items_list = [item for item in items_cursor]
        new_collection_name = '{}_batch#{}'.format(col_name, i)
        db[new_collection_name].insert_many(items_list)
        current_cols.append(new_collection_name)

    db.trendi_mongo_users.update_one({'name': username}, {'$set': {'collections': current_cols}})
