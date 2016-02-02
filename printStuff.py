__author__ = 'yonatan'
from time import sleep
import random
import subprocess

from termcolor import colored

colors = ["red", "blue", "green", "magenta", "yellow"]


if __name__ == "__main__":
    rc_list = []
    # for i in range(1):
    i = 1
    child = subprocess.Popen(["sudo ./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe"],
                             shell=True)
    r = random.randint(0, 4)
    print colored("BroseMe %s is opened" % (str(i)), colors[r])
    rc_list.append(child.returncode)
    a = 0
    print rc_list
    while all(rc is None for rc in rc_list):
        a += 1
        if a == 2:
            print colored("killllllllllllllllll", "green")
            c = subprocess.Popen("echo life is good")
            rc_list.append(c.returncode)
            print rc_list
            c.terminate()
            rc_list.append(c.returncode)
        sleep(10)

        print colored("still working", 'yellow')

    print colored("exiting", 'red')
    sleep(15)
