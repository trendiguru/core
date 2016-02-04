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
MAX_PER_DOMAIN = 20

whitelist = ["refinery29.com", "maxmodels.pl", "stylebistro.com", "fashion.ifeng.com", "tajbao.com",
             "highsnobiety.com", "osinka.ru", "trendsylvania.net", "whowhatwear.com", "fashiony.ru",
             "fashion.sina.com.cn", "lookbook.nu", "vogue.com.tw", "thefashionspot.com",
             "vogue.com.cn", "thehunt.com", "fashionbeans.com", "models.com",
             "fashion.sohu.com", "perfecte.md", "cosmopolitan.lt", "wwd.com", "enrz.com",
             "moteris.lt", "businessoffashion.com", "styleblazer.com", "theblondesalad.com", "fashiongonerogue.com",
             "thesartorialist.com", "cupcakesandcashmere.com", "fashion.walla.co.il", "thegloss.com", "vogue.com.au",
             "panele.lt", "af-110.com", "collegefashion.net", "niv.ru", "desired.de", "fashionstylemag.com",
             "guimi.com", "fashionbank.ru", "vmagazine.com", "garancedore.fr", "thefashionisto.com",
             "fashionising.com", "modelmanagement.com"]


def insertDomains():
    for domain in whitelist:
        item = {"name": domain,
                "locked": False,
                "paused": False,
                "last_processed": 0,
                "url_count": 1,
                "url_list": ["http://www." + domain]}
        db.scraped_urls.insert_one(item)


def screen(workers):
    workers = min(int(workers), len(whitelist))
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
        browseme = subprocess.Popen(["sudo ./xvfb-run-safe.sh python -m trendi.shakeNbake -f firefox &"],
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
                if link.startswith('/'):
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
    scr = open("/var/www/latest/b_main.js").read()
    while True:
        domains = db.scraped_urls.find({"locked": False, "paused": False})
        domains_count = domains.count()
        if domains_count > 0:
            rand = random.randint(0, domains_count - 1)
            domain = domains[rand]
            domain_id = domain["_id"]
            db.scraped_urls.update_one({"_id": domain_id}, {"$set": {"locked": True}})
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

            try:
                driver.get(url)
                print colored("got url %s with success" % url_printable, "cyan")
            except:
                print colored("failed getting url %s " % url_printable, "red", "on_yellow")
                db.scraped_urls.update_one({"_id": domain_id}, {"$set": {"locked": False,
                                                                             "last_processed": last_processed}})
                continue

            try:
                driver.set_page_load_timeout(2)
                elem = driver.find_element_by_xpath("//*")
                html = elem.get_attribute("outerHTML")
                print colored("got html with success on %s" % url_printable, "cyan")
            except:
                print colored("failed getting html on %s" % url_printable, "red", "on_yellow")
                db.scraped_urls.update_one({"_id": domain_id}, {"$set": {"locked": False,
                                                                         "last_processed": last_processed}})
                continue

            getAllUrls(url, html, domain_id)

            try:
                driver.set_script_timeout(5)
                response = driver.execute_async_script(scr)
                print colored("script executed! on %s" % url_printable, "blue", "on_green", attrs=['bold'])

                # for x in range(8):
                #     script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
                #     driver.execute_script(script)
                #     sleep(0.25)

            except:
                print colored("url %s : script execution failed!!!" % url_printable, "red", "on_yellow")

            db.scraped_urls.update_one({"_id": domain["_id"]}, {"$set": {"locked": False,
                                                                         "last_processed": last_processed}})
            driver.execute_script("window.stop();")
        else:
            all_domains = db.scraped_urls.find()
            updated = 0
            for domain in all_domains:
                url_count = domain["url_count"]
                current_count = len(domain["url_list"])
                if current_count > url_count:
                    url_count = current_count
                last_processed = domain["last_processed"]
                if last_processed < url_count and last_processed < MAX_PER_DOMAIN:
                    db.scraped_urls.update_one({"_id": domain["_id"]}, {"$set": {"paused": False, "locked": False}})
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
