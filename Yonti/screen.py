__author__ = 'yonatan'

import subprocess

from termcolor import colored


def master():
    print colored("starting master...", "green", attrs=["bold"])
    # commands = "./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe &"
    subprocess.call(
        ["screen -S scraper python -m trendi.printStuff"],
        shell=True)
    print colored("sraper detached/terminated", "green", attrs=["bold"])


if __name__ == "__main__":
    master()

