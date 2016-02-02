__author__ = 'yonatan'

from time import sleep
from os import system

from termcolor import colored


def master():
    print colored("starting master...", "green", attrs=["bold"])
    # commands = "./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe &"
    system(
        "screen -S test1 'sudo ./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe &'")
    sleep(2)
    print colored("closing master...", "green", attrs=["bold"])


if __name__ == "__main__":
    master()

