__author__ = 'yonatan'

import subprocess
from time import sleep
import argparse

from termcolor import colored


def getUserInput():
    parser = argparse.ArgumentParser(description='Main Scraper')
    parser.add_argument("func", default="screen", help=(
        "The function you want to run"))

    args = parser.parse_args()
    return args


def master():
    print colored("starting master...", "green", attrs=["bold"])
    # commands = "./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe &"
    subprocess.call(
        ["screen -S scraper python -m trendi.printStuff"],
        shell=True)
    print colored("sraper detached/terminated", "green", attrs=["bold"])


def runWorkers():
    rc_list = []
    tmpguard = subprocess.Popen(["python -m trendi.tmpGuard"], shell=True)
    rc_list.append(tmpguard)
    crawlme = subprocess.Popen(["rqworker -u redis://redis1-redis-1-vm:6379 CrawlMe"], shell=True)
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
    if user_input.func is "screen":
        master()
    elif user_input.func is "workers":
        runWorkers()
    else:
        print colored("bad input", "red")
