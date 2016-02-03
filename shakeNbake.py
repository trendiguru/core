__author__ = 'yonatan'

"""
all in one scraper
logic:
0. open screen with x processes
1. open browser
2. get url - check if domain in process - lock
3. connect to url
4. get html - send to scraper - which fills the url collection
5. inject js - scroll
6. do again from 2
"""

import subprocess
from time import sleep
import argparse

from bs4 import BeautifulSoup
from selenium import webdriver
from termcolor import colored

from . import constants

db = constants.db
MAX_PER_DOMAIN = 1000


def insertDomains():
    for domain in whitelist:
        item = {"name": domain,
                "locked": False,
                "paused": False,
                "last_processed": 0,
                "url_list": ["http://www." + domain]}
        db.scraped_urls.insert_one(item)


def screen(x):
    print colored("######  starting the scraper  ######", "green", attrs=["bold"])
    subprocess.call(
        ["screen -S scraper python -m trendi.shakeNbake -f processes -w ", x],
        shell=True)
    print colored("scraper detached/terminated", "green", attrs=["bold"])


def processes(x="1"):
    tmpguard = subprocess.Popen(["python -m trendi.tmpGuard"], shell=True)
    db.scraped_urls.delete_many({})
    insertDomains()
    for i in range(int(x)):
        browseme = subprocess.Popen(["sudo ./xvfb-run-safe.sh python -m trendi.shakeNbake -f firefox"],
                                    shell=True)
        print colored("firefox %s is opened" % (str(i)), 'green')

    subprocess.Popen(["screen -d scraper"], shell=True)

    while True:
        sleep(1000)


def getAllUrls(url, html, obid):
    soup = BeautifulSoup(html, "html.parser")
    domain = db.scraped_urls.find_one({"_id": obid})
    if domain:
        # find and process all the anchors in the document
        domain_name = domain["name"]
        url_list = domain["url_list"]
        url_count = len(url_list)
        for anchor in soup.find_all("a"):
            # extract link url from the anchor
            link = anchor.attrs["href"] if "href" in anchor.attrs else ''
            if not link.startswith(domain_name):
                if link.startswith('/'):
                    link = url + link
                else:
                    print ("link to a different site... not enqueued")
                    continue
            exists = [match for match in url_list if match == link]
            if len(exists) > 0:
                print colored("link already exists... ", "yellow")
            else:
                url_list.append(link)
        new_count = len(url_list)
        urls_added = new_count - url_count
        if urls_added > 0:
            db.scraped_urls.find_one_and_update({"_id": obid}, {"$set": {"url_list": url_list}})
        print colored("%s urls added to domain %s" % (str(urls_added), domain_name), "magenta", attrs=['bold'])


def firefox():
    driver = webdriver.Firefox()
    scr = open("/var/www/latest/b_main.js").read()
    while True:
        domain = db.scraped_urls.find_one_and_update({"locked": False, "paused": False}, {"$set": {"locked": True}})
        if domain:
            url_count = domain["url_count"]
            last_processed = domain["last_processed"]
            current_count = len(domain["url_list"])
            if current_count > url_count:
                url_count = current_count
            if last_processed >= url_count or last_processed > MAX_PER_DOMAIN:
                db.scraped_urls.update_one({"_id": domain["_id"]}, {"$set": {"paused": True}})
                continue
            url = domain["url_list"][last_processed]
            url_printable = url.encode('ascii', 'ignore')  # conversion of unicode type to string type
            last_processed += 1

            try:
                driver.get(url)
                print colored("got url %s with success" % url_printable, "green")
            except:
                print colored("failed getting url %s " % url_printable, "red", "on_yellow")
                db.scraped_urls.update_one({"_id": domain["_id"]}, {"$set": {"locked": False,
                                                                             "last_processed": last_processed}})
                continue

            elem = driver.find_element_by_css_selector('#my-id')
            html = elem.get_attribute('innerHTML')
            # subprocess.Popen(["python -m trendi.shakeNbake -f getAllUrls "], shell=True)
            getAllUrls(html, domain["_id"])

            try:
                driver.execute_script(scr)
                sleep(2)
                print colored("script executed!", "green")

                for x in range(8):
                    script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
                    driver.execute_script(script)
                    sleep(0.25)

            except:
                print colored("url %s : script execution failed!!!" % url_printable, "red", "on_yellow")

            db.scraped_urls.update_one({"_id": domain["_id"]}, {"$set": {"locked": False,
                                                                         "last_processed": last_processed}})
        else:
            all_domains = db.scraped_urls.find()
            updated = 0
            for domain in all_domains:
                url_count = domain["url_count"]
                current_count = len(domain["url_list"])
                if current_count > url_count:
                    url_count = current_count

                if last_processed < url_count and last_processed < MAX_PER_DOMAIN:
                    db.scraped_urls.update_one({"_id": domain["_id"]}, {"$set": {"paused": False}})
                    updated += 1
                    continue

            if updated == 0:
                exit()


def getUserInput():
    parser = argparse.ArgumentParser(description='Main Scraper')
    parser.add_argument("-f", dest="function", help="The function you want to run")
    parser.add_argument("-w", dest="workers", default="10", help="enter the number of workers to run simultaneously")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    user_input = getUserInput()
    print user_input
    if user_input.function == "screen":
        screen(user_input.workers)
    elif user_input.function == "processes":
        processes(user_input.workers)
    elif user_input.function == "firefox":
        firefox()
    else:
        print colored("bad input", "red")

whitelist = ["gettyimages.com", "tmz.com", "super.cz", "ew.com", "entretenimento.r7.com", "hollywoodlife.com",
             "kapanlagi.com", "zimbio.com", "jezebel.com", "purepeople.com", "jeanmarcmorandini.com"]
