__author__ = 'yonatan'
from time import sleep
import random

from termcolor import colored

colors = ["red", "blue", "green", "magenta", "yellow"]

if __name__ == "__main__":
    for i in range(30):
        sleep(1)
        r = random.randint(0, 5)
        print colored(colors[r], colors[r])
