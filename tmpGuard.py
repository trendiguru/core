__author__ = 'yonatan'

"""
this program checks the /tmp directory every 10 min
if the is an open webdriver that was not active in the last 10 min
it removes it.
the webdrivers are recognized by the 'tmp' opening
"""

import os
import time
import subprocess
import argparse

from termcolor import colored


def erasexvfb():
    xvfb2erase = []
    i = 0
    for f in os.listdir("/tmp"):
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat("/tmp/" + f)
        if f[:8] == "xvfb-run":
            xvfb2erase.append(f)
    return xvfb2erase


def erasetmp(cycle_delta=0, max_tmp=0):
    current = (time.ctime(time.time())).split(" ")
    current_date = current[-3]
    current_time = current[-2].split(":")
    current_hour = current_time[0]
    current_min = current_time[1]
    tmp2erase = []
    i = 0
    for f in os.listdir("/tmp"):
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat("/tmp/" + f)
        if f[:3] == "tmp":
            if max_tmp == 0:
                tmp2erase.append(f)
                continue
            i += 1
            modified = time.ctime(mtime)
            # print ("%s) %s  last modified: %s" % (str(i), f, modified))
            modified_list = modified.split(" ")
            modified_date = modified_list[-3]
            modified_time = modified_list[-2].split(":")
            modified_hour = modified_time[0]
            modified_min = modified_time[1]
            min_diff = int(current_min) - int(modified_min)
            hour_diff = int(current_hour) - int(modified_hour)
            date_diff = int(current_date) - int(modified_date)
            if date_diff == 0:
                if hour_diff == 0:
                    if min_diff > cycle_delta:
                        tmp2erase.append(f)
                elif hour_diff == 1:
                    if min_diff + 60 > cycle_delta:
                        tmp2erase.append(f)
                else:
                    tmp2erase.append(f)
            elif date_diff == 1:
                if hour_diff + 24 == 1:
                    if min_diff + 60 > cycle_delta:
                        tmp2erase.append(f)
                else:
                    tmp2erase.append(f)
            else:
                tmp2erase.append(f)
    if i > max_tmp and cycle_delta > 1:
        tmp2erase = erasetmp(1, 50)
    return tmp2erase


def mainDelete(filename, cycle=0, max_tmp=0):
    if filename == "tmp":
        files2erase = erasetmp(cycle, max_tmp)
    elif filename == "xvfb":
        files2erase = erasexvfb()
    else:
        print colored("bad input")
        exit(1)
    count = 0
    for f in files2erase:
        ret = subprocess.call(["sudo rm -r /tmp/" + f], shell=True)
        if not ret:
            print colored("removed %s succeeded" % f, "yellow")
            count += 1
        else:
            print colored("removing %s failed" % f, "red")

    print colored("%s files deleted!!!" % str(count))
    # time.sleep(15)


def getUserInput():
    parser = argparse.ArgumentParser(description='Main tmpGuard')
    parser.add_argument("-m", dest="max", default="0", help="The max number of tmp allowed")
    parser.add_argument("-d", dest="delta", default="0", help="last modified delta")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    user_input = getUserInput()
    print user_input
    mainDelete("xvfb")
    delta = int(user_input.delta)
    maxTmp = int(user_input.max)
    while True:
        mainDelete("tmp", cycle=delta, max_tmp=maxTmp)
        if maxTmp == 0:
            break
        time.sleep(300)
    exit(0)
