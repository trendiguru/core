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

from termcolor import colored


def main():
    current = (time.ctime(time.time())).split(" ")
    print current
    current_date = current[-3]
    current_time = current[-2].split(":")
    current_hour = current_time[0]
    current_min = current_time[1]
    files2erase = []
    for file in os.listdir("/tmp"):
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat("/tmp/" + file)
        if file[:3] == "tmp":
            modified = time.ctime(mtime)
            print (file + "last modified: %s" % modified)
            modified_list = modified.split(" ")
            modified_date = modified_list[-3]
            modified_time = modified_list[-2].split(":")
            modified_hour = modified_time[0]
            modified_min = modified_time[1]
            min_diff = int(modified_min) - int(current_min)
            hour_diff = int(modified_hour) - int(current_hour)
            date_diff = int(modified_date) - int(current_date)
            if date_diff == 0:
                if hour_diff == 0:
                    if min_diff > 10:
                        files2erase.append(file)
                elif hour_diff == 1:
                    if min_diff + 60 > 10:
                        files2erase.append(file)
                else:
                    files2erase.append(file)
            elif date_diff == 1:
                if hour_diff + 24 == 1:
                    if min_diff + 60 > 10:
                        files2erase.append(file)
                else:
                    files2erase.append(file)
            else:
                files2erase.append(file)
    count = 0
    for file in files2erase:
        ret = subprocess.call(["sudo rm -r /tmp/" + file], shell=True)
        if not ret:
            print colored("removed X succeeded", "yellow")
            count += 1
        else:
            print colored("removing X failed", "red")

    print colored("%s files deleted!!!" % str(count))


if __name__ == "__main__":
    main()
