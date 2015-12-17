__author__ = 'yonatan'

import sys

from bs4 import BeautifulSoup
import requests
import requests.exceptions
from rq import Queue

from browseme import runExt
import constants

browse_q = Queue('BrowseMe', connection=constants.redis_conn)
scrap_q = Queue('CrawlMe', connection=constants.redis_conn)
db = constants.db
theLobby = 0


def scrapLinks(url, floor):
    print("Crawling %s" % url)
    exists = db.crawler_processed.find_one({"url": url})
    if exists:
        print ("url already exists... scraper skips")
        return
    db.crawler_processed.insert_one({"url": url})
    if floor is not theLobby:
        floor -= 1
        # get url's content
        try:
            response = requests.get(url)
        except (requests.exceptions.MissingSchema, requests.exceptions.ConnectionError):
            # ignore pages with errors
            print("Crawl fail for %s" % url)
            return
        # create a beutiful soup for the html document
        soup = BeautifulSoup(response.text, "html.parser")

        # find and process all the anchors in the document
        for anchor in soup.find_all("a"):
            # extract link url from the anchor
            link = anchor.attrs["href"] if "href" in anchor.attrs else ''
            if not link.startswith('http'):
                link = url + link
            exists = db.crawler_processed.find_one({"url": link})
            if exists:
                print ("new link already exists... not enqueued")
            else:
                scrap_q.enqueue(scrapLinks, link, floor)

    browse_q.enqueue(runExt, url)
    return


def masterCrawler(floor=2):
    db.crawler_processed.drop()
    db.crawler_processed.create_index("url")
    for site in constants.white_list:
        scrap_q.enqueue(scrapLinks, site, floor)
    return "finished"


if __name__ == "__main__":
    print ("Scraping the white list - Started...)")
    floor = 2
    if len(sys.argv) == 2:
        floor = int(sys.argv[1])
    res = masterCrawler(floor)
    print (res)
