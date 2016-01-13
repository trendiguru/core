
from rq import Queue
from trendi import constants

con = constants.redis_conn
bl = Queue('blacklisted_urls', connection = con)
dicts = bl.find()
for dict in dicts:
    url =  dict['page_url']
    print('url:'+url)