
from rq import Queue
from trendi import constants

db = constants.db
bl = db.blacklisted_urls
dicts = bl.find()
for dict in dicts:
    url =  dict['page_url']
    print('url:'+url)