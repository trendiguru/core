__author__ = 'Nadav Paz'

import time
import datetime

from . import monitoring
from . import crawlme
from . import constants


redis_conn = constants.redis_conn
from rq import Queue

browseMe_q = Queue('BrowseMe', redis_conn)
crawlMe_q = Queue('CrawlMe', redis_conn)

if __name__ == "__main__":
    print "start time: {0}".format(datetime.datetime.utcnow())
    link_list = ['http://www.' + domain[3:] for domain in monitoring.get_top_x_whitelist(100).keys()]
    for link in link_list:
        while crawlMe_q.count > 1000:
            time.sleep(300)
        print "Gonna scrape {0} now..".format(link)
        crawlme.scrapLinks(link, 2)
    while browseMe_q.count > 20:
        time.sleep(30)
    print "end at {0}".format(datetime.datetime.utcnow())