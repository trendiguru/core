__author__ = 'yonatan'
from time import sleep
import random
import subprocess

from termcolor import colored

colors = ["red", "blue", "green", "magenta", "yellow"]

if __name__ == "__main__":
    for i in range(10):
        sleep(1)
        subprocess.call(["sudo ./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe &"], shell=True)
        r = random.randint(0, 4)
        print colored(str(i) + str(i) + str(i), colors[r])
