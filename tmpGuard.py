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


def main(manual=True):
    current = (time.ctime(time.time())).split(" ")
    current_date = current[-3]
    current_time = current[-2].split(":")
    current_hour = current_time[0]
    current_min = current_time[1]
    # tmp_min = int(current_min)
    # print type(tmp_min)
    # if divmod(tmp_min, 10)[1] != 0 and manual:
    #     print colored("exited without deleting - minute isn't dividable by 10")
    #     return
    # print colored("\ntmpGuard : %s\n" % current[-2], "red", "on_yellow")
    files2erase = []
    i = 0
    for f in os.listdir("/tmp"):
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat("/tmp/" + f)
        if f[:3] == "tmp":
            i += 1
            modified = time.ctime(mtime)
            print ("%s) %s  last modified: %s" % (str(i), f, modified))
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
                    if min_diff > 5:
                        files2erase.append(f)
                elif hour_diff == 1:
                    if min_diff + 60 > 5:
                        files2erase.append(f)
                else:
                    files2erase.append(f)
            elif date_diff == 1:
                if hour_diff + 24 == 1:
                    if min_diff + 60 > 5:
                        files2erase.append(f)
                else:
                    files2erase.append(f)
            else:
                files2erase.append(f)
    count = 0
    for f in files2erase:
        ret = subprocess.call(["sudo rm -r /tmp/" + f], shell=True)
        if not ret:
            print colored("removed %s succeeded" % f, "yellow")
            count += 1
        else:
            print colored("removing %s failed" % f, "red")

    print colored("%s files deleted!!!" % str(count))


if __name__ == "__main__":
    while True:
        main(False)
        time.sleep(300)
