__author__ = 'yonatan'
from time import sleep
import subprocess

from termcolor import colored

if __name__ == "__main__":
    rc_list = []
    crawlme = subprocess.Popen(["rqworker -u redis://redis1-redis-1-vm:6379 CrawlMe"], shell=True)
    rc_list.append(crawlme)
    for i in range(10):
        browseme = subprocess.Popen(["sudo ./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe"],
                                    shell=True)
        print colored("BroseMe %s is opened" % (str(i)), green)
        rc_list.append(browseme)

    while all(rc.returncode is None for rc in rc_list):
        sleep(60)
        print colored("still working", 'yellow')

    print colored("exiting", 'red')
    sleep(15)
