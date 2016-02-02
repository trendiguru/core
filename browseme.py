__author__ = 'yonatan'
import time
import random
import os

from termcolor import colored
from selenium import webdriver
from selenium.webdriver.common.proxy import *
from rq import Queue
from redis import Redis

from . import constants

db = constants.db
redis_conn = Redis(host="redis1-redis-1-vm")
person_job_Q = Queue("person_job", connection=redis_conn)
# paperdoll_Q = Queue("pd", connection=redis_conn)
browse_q = Queue('BrowseMe', connection=redis_conn)


def getProxy():
    pro = random.sample(proxies, 1)
    myProxy = pro[0][0] + ":" + pro[0][1]
    print colored('using Proxy = ' + myProxy, 'magenta', attrs=['bold'])
    proxy = Proxy({
        'proxyType': ProxyType.MANUAL,
        'httpProxy': myProxy,
        'ftpProxy': myProxy,
        'sslProxy': myProxy,
        'noProxy': None,
        'autodetect': False
    })

    return proxy


def checkDomain(domain):
    domain_exists = db.scraped_urls.find({"$and": [{"domain": domain},
                                                   {"domain_locked": True}]})
    if domain_exists.count() > 2:
        return True
    return False


def runExt(url, domain):
    print colored("Running Extension on %s" % url, "magenta", attrs=['bold'])
    domain_locked = checkDomain(domain)
    if domain_locked:
        print colored("domain is processed in this moment  - returning to Q", "yellow")
        browse_q.enqueue(runExt, url, domain)
        exit()
    else:
        db.scraped_urls.update_one({"url": url}, {"$set": {"domain_locked": True}})

    driver = webdriver.Firefox()
    try:
        scr = open("/var/www/latest/b_main.js").read()
        print colored("driver started", "yellow")
        # wait for the queues to be empty enough
        countQue = 0
        while person_job_Q.count > 500:
            countQue += 1
            if countQue > 2:
                print colored("person job Que Full - returned to Que", "green", attrs=['bold'])
                browse_q.enqueue(runExt, url, domain)
                driver.quit()
                exit()
            print colored("person job Que Full - taking 15 sec break", "red")
            time.sleep(15)
        # print colored("0", "green")
        driver.get(url)
        # print colored("1", "green")
        driver.execute_script(scr)
        # print colored("2", "green")
        time.sleep(1)
        print colored("script executed!", "green")

        for x in range(8):
            script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
            driver.execute_script(script)
            time.sleep(0.25)
        print colored("execute Success!!!", "yellow", "on_magenta", attrs=['bold'])
    except:
        print colored("execute Failed!!!", "red", "on_yellow", attrs=['bold'])

    db.scraped_urls.update_one({"url": url}, {"$set": {"domain_locked": False}})
    try:
        driver.quit()
    except:
        print colored("driver.quit() Failed!!!", "red", "on_yellow", attrs=['bold'])


proxies = [['118.142.33.112', '8088'],
           ['31.173.74.73', '8080'],
           ['198.169.246.30', '80'],
           ['202.29.97.2', '3128'],
           ['91.121.181.168', '80']]


def get_count(start_path='.'):
    repeat = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # total_size += os.path.getsize(fp)
            print f
            if f[:3] == 'tmp':
                repeat += 1
    return repeat
