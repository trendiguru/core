__author__ = 'Nadav Paz'

import time
import datetime

from . import monitoring
from . import crawlme
from . import constants

redis_conn = constants.redis_conn
from rq import Queue
from . import mainScraper


browseMe_q = Queue('BrowseMe', connection=redis_conn)
crawlMe_q = Queue('CrawlMe', connection=redis_conn)

if __name__ == "__main__":
    urlid = time.time()
    print "start time: {0}".format(datetime.datetime.utcnow())
    if not mainScraper.screenCheck():
        mainScraper.master()
    link_list = ['http://www.' + domain[4:] for domain in monitoring.get_top_x_whitelist(100).keys()]
    for link in link_list:
        while crawlMe_q.count > 1000:
            time.sleep(300)
        print "Gonna scrape {0} now..".format(link)
        crawlme.scrapLinks(link, urlid, 2)
    while browseMe_q.count > 20:
        time.sleep(30)
    print "end at {0}".format(datetime.datetime.utcnow())