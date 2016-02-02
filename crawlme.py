__author__ = 'yonatan'

from bs4 import BeautifulSoup
import requests
import requests.exceptions
from rq import Queue
from termcolor import colored

from .browseme import runExt
import constants

browse_q = Queue('BrowseMe', connection=constants.redis_conn)
scrap_q = Queue('CrawlMe', connection=constants.redis_conn)
db = constants.db
theLobby = 0


def scrapLinks(url, runid, floor, domain=None):
    domain = domain or url
    url_printable = url.encode('ascii', 'ignore')  # conversion of unicode type to string type
    print colored("Crawling %s" % url_printable, "yellow")
    url_exists = db.scraped_urls.find_one({"url": url})
    if url_exists:
        if url_exists["runId"] == runid:
            print ("url already exists... scraper skips")
            return
        else:
            db.scraped_urls.update_one({"_id": url_exists["_id"]}, {"$set": {"runId": runid}})
    else:
        new_url = {"domain": domain,
                   "domain_locked": False,
                   "url": url,
                   "runId": runid}
        db.scraped_urls.insert_one(new_url)
    if floor == theLobby:
        pass
    else:
        floor -= 1
        # get url's content
        try:
            response = requests.get(url)
        except (requests.exceptions.MissingSchema, requests.exceptions.ConnectionError):
            # ignore pages with errors
            url_printable = url.encode('ascii', 'ignore')  # conversion of unicode type to string type
            print colored("Crawl fail for %s" % url_printable, "red")
            return
        # create a beutiful soup for the html document
        soup = BeautifulSoup(response.text, "html.parser")

        # find and process all the anchors in the document
        for anchor in soup.find_all("a"):
            # extract link url from the anchor
            link = anchor.attrs["href"] if "href" in anchor.attrs else ''
            if not link.startswith(domain):
                if link.startswith('/'):
                    link = url + link
                else:
                    print ("link to a different site... not enqueued")
                    continue
            exists = db.scraped_urls.find_one({"url": link})
            if exists:
                print ("link already exists... not enqueued")
            else:
                scrap_q.enqueue(scrapLinks, link, runid, floor, domain)
    print colored("%s sent to BrowseMe" % url_printable, "green")
    browse_q.enqueue(runExt, url, domain)
    return



