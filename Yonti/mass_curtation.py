from ..constants import db


def create_collection_for_main_category(source_col, category, user_name):
    tmp_col = 'zzz_JUNK_zzz'
    output_col = '%s_%s' %(user_name,category)
    db[source_col].aggregate([{'$match':{'categories':category}}, {'$out': tmp_col}])
    db[tmp_col].copyTo(output_col)
    print ('new collection : %s was created/updated!\n'
           'collection contains %d items under %s category' % (output_col, db[output_col].count(), category))


def reset_evaluation_keys(col_name):
    collection = db[col_name]
    evaluation = {'checker1':False,
            'checker2':False,
            'checker3':False}
    collection.update_many({},{'$set':{'evaluation': evaluation}})
    print ('reset evaluation key Done!')


def update_checker_status(current_status):
    updated_status = current_status
    for key in current_status.keys():
        if not current_status[key]:
            updated_status[key]=True
            break
    return updated_status


def pull_candidates(col_name,checkers_status,batch_size):
    collection = db[col_name]
    res= []
    ids = []
    for item in collection.find({'evaluation':checkers_status},{'images.Xlarge':1}).limit(batch_size*2):
        res.append(item)
        ids.append(item['_id'])

    collection.update_many({'_id':{'$in':ids}}, {'$set':{'evaluation':update_checker_status(checkers_status)}})

    return res


def shoot_dont_talk(col_name, to_delete, checkers_status, batch_size=50):
    '''
    deletes bad labeles imgs
    fetch next batch
    '''
    collection = db[col_name]
    if len(to_delete):
        collection.delete_many({'_id':{'$in':to_delete}})

    return pull_candidates(col_name,checkers_status,batch_size)

