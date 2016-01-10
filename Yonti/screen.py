__author__ = 'yonatan'

import subprocess
from time import sleep

from termcolor import colored


def master():
    print colored("starting master...", "green", attrs=["bold"])
    subprocess.call(["screen -S test1 'python printStuff.py'"], shell=True)
    sleep(2)
    print colored("closing master...", "green", attrs=["bold"])


if __name__ == "__main__":
    master()

