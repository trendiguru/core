import pickle

import tldextract
from rq import Queue

from . import constants

scrap_q = Queue('CrawlMe', connection=constants.redis_conn)
db = constants.db
WHITELIST_PATH = '/home/www-data/whitelist.txt'


def add_to_whitelist(url):
    with open(WHITELIST_PATH, 'rb') as f:
        wl = pickle.load(f)
    try:
        reg = tldextract.extract(url).registered_domain
        if reg not in wl:
            wl.append(reg)
            with open(WHITELIST_PATH, 'wb') as f:
                pickle.dump(wl, f)
            return "Done! :)"
        else:
            return reg + " is already in the WhiteList! :)"
    except:
        return "Failed for some reason.. :( check url"


    # def masterCrawler(floor=2, whiteList=top50Fashion):
    # db.scraped_urls.delete_many()

    # db.scraped_urls.create_indexes(["url", "domain", "urlId", "domain_locked"])

# urlid = time.time()
#     for site in whiteList:
#         url = "http://www." + site
#         scrap_q.enqueue(scrapLinks, url, urlid, floor)
#     return "finished"
#
#
# if __name__ == "__main__":
#     print ("Scraping the white list - Started...)")
#     levels = 2
#     whiteLi = top50Fashion
#     if len(sys.argv) > 1:
#         levels = int(sys.argv[1])
#     if len(sys.argv) > 2:
#         if sys.argv[2] == "top50CelebSytle":
#             whiteLi = top50CelebSytle
#         elif sys.argv[2] == "fashionBlogs":
#             whiteLi = fashionBlogs
#         elif sys.argv[2] == "fullList":
#             whiteLi = fullList
#         else:
#             whiteLi = [sys.argv[2]]
#     res = masterCrawler(levels, whiteLi)
#     print (res)
