
from rq import Queue
from trendi import constants

db = constants.db
bl = db.blacklisted_urls
dicts = bl.find()
print('count:'+str(dicts.count()))
raw_input()
dict = dicts.next()
while dict is not None:
    url =  dict['page_url']
    print('url:'+url)
    dict = dicts.next()
