__author__ = 'yonatan'

import subprocess
from time import sleep
import argparse

from selenium import webdriver
from termcolor import colored
import pymongo

from . import tmpGuard

db = pymongo.MongoClient(host="mongodb1-instance-1", port=27017).mydb
MAX_PER_DOMAIN = 1

url = ['http://lookbook.nu/']


def screen(workers):
    print colored("######  starting the scraper  ######", "green", attrs=["bold"])
    db.scraped_urls.delete_many({})
    tmpGuard.mainDelete("xvfb")
    tmpGuard.mainDelete("tmp")
    cmd = "screen -S scraper python -m trendi.Yonti.oneSiteScraper -f processes -w " + str(workers)
    print colored("opening screen", "green", attrs=["bold"])
    subprocess.call([cmd], shell=True)
    print colored("screen detached", "yellow", attrs=["bold"])


def processes(w):
    sleep(1)
    for i in range(int(w)):
        sleep(1)
        browseme = subprocess.Popen(["sudo ./xvfb-run-safe.sh python -m trendi.Yonti.oneSiteScraper -f firefox"],
                                    shell=True)
        print colored("firefox %s is opened" % (str(i)), 'green')

    sleep(60)
    # get permission for tmp files
    subprocess.call(["sudo chmod -R /tmp/tmp*" ], shell=True)
    sleep(5)
    ret1 = subprocess.call(["sudo rm -r /tmp/tmp*/cache2/entries/" ], shell=True)
    ret2 = subprocess.call(["sudo rm -r /tmp/tmp*/thumbnails/" ], shell=True)
    if not ret1 or not ret2 :
        print colored("removed succeeded", "yellow")
    else:
        print colored("removing failed", "red")
    while True:
        sleep(1000)


def firefox():
    driver = webdriver.Firefox()
    driver.implicitly_wait(10)
    # driver.set_page_load_timeout(2)

    scr = open("/var/www/latest/run_ext.js").read()
    while True:
        try:
            driver.get(url)
            # print colored("got url %s with success" % url_printable, "cyan")
        except:
            print colored("URL failed on %s" % url, "blue", "on_yellow")
            sleep(30)
            continue

        try:
            driver.execute_script(scr)
            print colored("sctipt executed! ", "blue", "on_green", attrs=['bold'])
            sleep(5)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(5)

            # for x in range(40):
            #     script = "scroll(" + str(x * 50) + "," + str(x * 50 + 50) + ")"
            #     driver.execute_script(script)
            #     sleep(0.05)

        except:
            print colored("EXECUTE failed on %s " % url, "red", "on_yellow")

        sleep(50)
    driver.quit()


def getUserInput():
    parser = argparse.ArgumentParser(description='Main Scraper')
    parser.add_argument("-f", dest="function", default="screen", help="The function you want to run")
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
