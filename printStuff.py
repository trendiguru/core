__author__ = 'yonatan'
from time import sleep
import random
import subprocess

from termcolor import colored

colors = ["red", "blue", "green", "magenta", "yellow"]

if __name__ == "__main__":
    rc_list = []
    for i in range(2):
        child = subprocess.Popen(["sudo ./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe"],
                                 shell=True)
        r = random.randint(0, 4)
        print colored("BroseMe %s is opened" % (str(i)), colors[r])
        rc_list.append(child.returncode)

    while all(rc is None for rc in rc_list):
        sleep(30)
        print colored("still working", 'yellow')

    print colored("exiting", 'red')
    sleep(15)
