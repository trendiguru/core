__author__ = 'yonatan'

import subprocess
from time import sleep

from termcolor import colored


def master():
    print colored("starting master...", "green", attrs=["bold"])
    # commands = "./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe &"
    subprocess.call(
        ["screen -S test1 sudo bash ./xvfb-run-safe.sh rqworker -u redis://redis1-redis-1-vm:6379 BrowseMe &"],
                    shell=True)
    sleep(2)
    print colored("closing master...", "green", attrs=["bold"])


if __name__ == "__main__":
    master()

