__author__ = 'yonatan'

import subprocess
from time import sleep
import argparse

from termcolor import colored

import whitelist


def getUserInput():
    parser = argparse.ArgumentParser(description='Main Scraper')
    parser.add_argument("func", default="screen", help=(
        "The function you want to run"))
    parser.add_argument("-scrap", "-s", dest="list2scrap", default="None", help=(
        "enter list name if you want to scrap a list from whitelist.py"))
    parser.add_argument("-floors", "-f", dest="floors", default="2", help=(
        "enter how many pages in to scrap"))
    args = parser.parse_args()
    return args


def master():
    print colored("starting master...", "green", attrs=["bold"])
    subprocess.call(
        ["screen -S scraper python -m trendi.mainScraper workers"],
        shell=True)
    print colored("scraper detached/terminated", "green", attrs=["bold"])


def runWorkers():
    rc_list = []
    tmpguard = subprocess.Popen(["python -m trendi.tmpGuard"], shell=True)
    rc_list.append(tmpguard)
    crawlme = subprocess.Popen(["rqworker -u redis://redis1-redis-1-vm:6379 CrawlMe "], shell=True)
    rc_list.append(crawlme)
    for i in range(10):
        browseme = subprocess.Popen(["sudo ./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe"],
                                    shell=True)
        print colored("BroseMe %s is opened" % (str(i)), 'green')
        rc_list.append(browseme)

    while all(rc.returncode is None for rc in rc_list):
        sleep(60)
        print colored("still working", 'yellow')

    print colored("exiting", 'red')
    sleep(15)


if __name__ == "__main__":
    user_input = getUserInput()
    print user_input
    if user_input.func == "screen":
        if user_input.list2scrap != "None":
            try:
                floor = int(user_input.floors)
            except:
                print colored("floor is not an int", "red")
                exit(1)
            whitelist.masterCrawler(floor)
        master()
    elif user_input.func == "workers":
        runWorkers()
    else:
        print colored("bad input", "red")
