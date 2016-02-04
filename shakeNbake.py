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
import random

from bs4 import BeautifulSoup
from selenium import webdriver
from termcolor import colored
import pymongo

from . import tmpGuard

db = pymongo.MongoClient(host="mongodb1-instance-1", port=27017).mydb
MAX_PER_DOMAIN = 1000

whitelist = ["gettyimages.com", "manrepeller.com", "wishwishwish.net", "parkandcube.com", "stellaswardrobe.com",
             "cocosteaparty.com",
             "5inchandup.blogspot.co.uk", "garypeppergirl.com", "camilleovertherainbow.com", "streetpeeper.com",
             "the-frugality.com", "disneyrollergirl.net", "weworewhat.com", "wearingittoday.co.uk",
             "ella-lapetiteanglaise.com",
             "advancedstyle.blogspot.co.uk", "indtl.com", "redcarpet-fashionawards.com", "nadiaaboulhosn.com",
             "enbrogue.com",
             "peonylim.com", "vanessajackman.blogspot.co.uk", "alltheprettybirds.com", "lisegrendene.com.br",
             "nataliehartleywears.blogspot.co.uk", "tommyton.com", "stylebubble.co.uk", "pandorasykes.com",
             "theblondesalad.com", 'notorious-mag.com',
             "thesartorialist.com", "bryanboy.com", "bunte.de", "gala.fr",
             "pudelek.pl", "tmz.com", "super.cz", "ew.com", "entretenimento.r7.com", "hollywoodlife.com",
             "kapanlagi.com", "zimbio.com", "jezebel.com", "purepeople.com", "jeanmarcmorandini.com",
             "radaronline.com", "etonline.com", "voici.fr", "topito.com", "ciudad.com.ar", "perezhilton.com",
             "koreaboo.com", "cztv.com", "virgula.uol.com.br", "suggest.com", "justjared.com", "therichest.com",
             "pressroomvip.com", "dagospia.com", "closermag.fr", "kiskegyed.hu", "pagesix.com", "spynews.ro",
             "digitalspy.com", "purepeople.com.br", "thepiratebay.uk.net", "sopitas.com", "deadline.com",
             "starpulse.com", "multikino.pl", "zakzak.co.jp", "primiciasya.com", "celebuzz.com", "luckstars.co",
             "ratingcero.com", "non-stop-people.com", "tochka.net", "toofab.com", "extra.cz", "kozaczek.pl",
             "huabian.com", "bossip.com", "spletnik.ru", "wetpaint.com"]


def insertDomains():
    items = []
    for domain in whitelist:
        item = {"name": domain,
                "locked": False,
                "paused": False,
                "last_processed": 0,
                "url_count": 1,
                "url_list": ["http://www." + domain]}
        items.append(item)
    db.scraped_urls.insert_many(items)


def screen(workers):
    # workers = min(int(workers), len(whitelist))
    print colored("######  starting the scraper  ######", "green", attrs=["bold"])
    db.scraped_urls.delete_many({})
    tmpGuard.mainDelete("xvfb")
    tmpGuard.mainDelete("tmp")
    insertDomains()
    cmd = "screen -S scraper python -m trendi.shakeNbake -f processes -w " + str(workers)
    print colored("opening screen", "green", attrs=["bold"])
    subprocess.call([cmd], shell=True)
    print colored("screen detached", "yellow", attrs=["bold"])


def processes(w):
    for i in range(int(w)):
        browseme = subprocess.Popen(["sudo ./xvfb-run-safe.sh python -m trendi.shakeNbake -f firefox"],
                                    shell=True)
        print colored("firefox %s is opened" % (str(i)), 'green')

    sleep(5)
    subprocess.Popen(["screen -d"], shell=True)

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
                if link.startswith('/') and len(link) > 3:
                    link = url + link
                else:
                    # print ("link to a different site... not enqueued")
                    continue
            exists = [match for match in url_list if match == link]
            if len(exists) > 0:
                # print colored("link already exists... ", "yellow")
                pass
            else:
                url_list.append(link)
        new_count = len(url_list)
        urls_added = new_count - url_count
        if urls_added > 0:
            db.scraped_urls.find_one_and_update({"_id": obid}, {"$set": {"url_list": url_list, "url_count": new_count}})
        print colored("%s urls added to domain %s, url count for this domain is %s " % (str(urls_added), domain_name,
                                                                                        str(new_count)), "magenta",
                      attrs=['bold'])


def firefox():
    driver = webdriver.Firefox()
    driver.implicitly_wait(10)
    # driver.set_page_load_timeout(2)

    scr = open("/var/www/latest/b_main.js").read()
    while True:
        domains = db.scraped_urls.find({"locked": False, "paused": False})
        domains_count = domains.count()
        if domains_count > 0:
            rand = random.randint(0, domains_count - 1)
            try:
                domain = domains[rand]
            except:
                continue
            if domain["locked"]:
                continue
            domain_id = domain["_id"]
            url_count = domain["url_count"]
            last_processed = domain["last_processed"]
            current_count = len(domain["url_list"])
            if current_count > url_count:
                url_count = current_count
            if last_processed >= url_count or last_processed > MAX_PER_DOMAIN:
                db.scraped_urls.update_one({"_id": domain_id}, {"$set": {"paused": True}})
                print colored("domain %s is paused!!! " % domain["name"], "yellow")
                continue

            url = domain["url_list"][last_processed]
            url_printable = url.encode('ascii', 'ignore')  # conversion of unicode type to string type
            last_processed += 1
            db.scraped_urls.update_one({"_id": domain_id}, {"$set": {"locked": True, "last_processed": last_processed}})
            try:
                driver.get(url)
                # print colored("got url %s with success" % url_printable, "cyan")
            except:
                print colored("URL failed on %s" % url_printable, "blue", "on_yellow")
                db.scraped_urls.update_one({"_id": domain_id}, {"$set": {"locked": False}})

                continue

            try:
                # driver.set_page_load_timeout(2)
                elem = driver.find_element_by_xpath("//*")
                html = elem.get_attribute("outerHTML")
                # print colored("got html with success on %s" % url_printable, "cyan")
                getAllUrls(url, html, domain_id)
            except:
                print colored("HTML failed on %s" % url_printable, "blue", "on_yellow")
                # db.scraped_urls.update_one({"_id": domain_id}, {"$set": {"locked": False}})
                # # driver.execute_script("window.stop();")
                # continue

            try:
                # driver.set_script_timeout(1)
                driver.execute_script(scr)
                print colored("script executed! on %s" % url_printable, "blue", "on_green", attrs=['bold'])
                sleep(2)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                sleep(2)
                #
                # for x in range(40):
                #     script = "scroll(" + str(x * 50) + "," + str(x * 50 + 50) + ")"
                #     driver.execute_script(script)
                #     sleep(0.05)

            except:
                print colored("EXECUTE failed on %s " % url_printable, "red", "on_yellow")

            db.scraped_urls.update_one({"_id": domain["_id"]}, {"$set": {"locked": False}})
            # driver.execute_script("window.stop();")

        else:
            # check if it because all are locked
            domains = db.scraped_urls.find({"locked": False})
            domains_count = domains.count()
            if domains_count == 0:
                sleep(2)
                continue
            # check for jams
            all_domains = db.scraped_urls.find()
            updated = 0
            for domain in all_domains:
                url_count = domain["url_count"]
                current_count = len(domain["url_list"])
                if current_count > url_count:
                    url_count = current_count
                last_processed = domain["last_processed"]
                if last_processed < url_count and last_processed < MAX_PER_DOMAIN:
                    db.scraped_urls.update_one({"_id": domain["_id"]}, {"$set": {"paused": False}})
                    print colored("domain %s is resumed!!! " % domain["name"], "yellow")
                    updated += 1
                    continue

            if updated == 0:
                break

    driver.quit()


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
