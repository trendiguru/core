import subprocess
from time import sleep
import argparse
import sys
from termcolor import colored


def screen(workers):
    print colored("######  starting to open FPs  ######", "green", attrs=["bold"])
    cmd = "screen -S manyMore python -m trendi.db_stuff.manyMoreFPs -f fps -n " + str(workers)
    sleep(2)
    print colored("opening screen", "green", attrs=["bold"])
    sleep(2)
    subprocess.call([cmd], shell=True)
    print colored("screen detached", "yellow", attrs=["bold"])


def fps(w):
    print("many more fps")
    for i in range(int(w)):
        sleep(1)
        a = subprocess.Popen(["sudo rqworker -u redis://redis1-redis-1-vm:6379 fingerprint_new"], shell=True)
        print colored("fp_new #%s is opened" % (str(i)), 'green')

    while True:
        sleep(1000)


def get_user_input():
    parser = argparse.ArgumentParser(description='give me some more fingerprinters!')
    parser.add_argument("-n", dest="workers", default="25", help="enter the number of workers to run simultaneously")
    parser.add_argument("-f", dest="function", default="screen", help="which function 2 run")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    user_input = get_user_input()
    if user_input.function == "screen":
        screen(user_input.workers)
    elif user_input.function == "fps":
        fps(user_input.workers)
    else:
        print("bad input")
    sys.exit(0)

