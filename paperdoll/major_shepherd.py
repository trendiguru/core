__author__ = 'jeremy'
import subprocess, signal
import time
import os
import socket
import argparse

from trendi import constants
import psutil
import random


def kill_all_workers():
    #full command to start worker is:
 #   cd /home/jeremy/paperdoll/usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
    p = subprocess.Popen(['ps', '-auxw'], stdout=subprocess.PIPE)
    out, err = p.communicate()
#break ps output down into lines and loop on them...:
    string_to_look_for_in_rq_command = constants.string_to_look_for_in_rq_command
    for line in out.splitlines():
        if string_to_look_for_in_rq_command in line:
            a = line.split()
            pid = int(a[1])  #maybe on a different unix the output doesnt have owqnder
            print('pid to kill:'+str(pid))
#            pid = int(line.split(None, 1)[1])
            r = os.kill(pid, signal.SIGKILL)
            print r

def count_queue_workers(unique_string):
    n = 0
    #full command to start worker is:
 #   /usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
#    command = 'ps -auxw'
 #   p = subprocess.Popen(command, shell=True,stdout=subprocess.PIPE).stdout.read()
    string_to_look_for_in_rq_command = constants.string_to_look_for_in_rq_command
    p = subprocess.Popen(['ps', '-aux'], stdout=subprocess.PIPE)
    out, err = p.communicate()
#break ps output down into lines and loop on them...:
    for line in out.splitlines():
        #print line
        if string_to_look_for_in_rq_command in line and unique_string in line:
            a = line.split()
            print a
            pid = a[1]
            n = n +1
    n = n/2  #there is adouble count of each process, one appears with /bin/sh at beginning of ps line
    return n

def start_workers(command,n_workers):
    host = socket.gethostname()
    print('host:'+str(host)+' trying to start '+str(n_workers)+' workers with command '+str(command))
    for i in range(0,n_workers):
        print('command:'+command)
        p = subprocess.Popen(command, shell=True,stdout=subprocess.PIPE)
        print(p)

def restart_workers():
    kill_all_workers()
    time.sleep(2)
    for i in range(0,len(constants.worker_commands)):
        n_workers = constants.N_expected_workers[i]
        command = constants.worker_commands[i]
        start_workers(command,n_workers)
        time.sleep(10)

def kill_worker(unique = 'find_similar'):
    p = subprocess.Popen(['ps', '-auxw'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    pids = []
    for line in out.splitlines():
        if unique in line:
            a = line.split()
            pid = int(a[1])  #maybe on a different unix the output doesnt have owqnder
            print('a pid to kill:'+str(pid)+' using unique string:'+str(unique))
#            pid = int(line.split(None, 1)[1])
            pids.append(pid)
    n=random.randrange(len(pids))
    print('{0} got the short straw'.format(pids[n]))
    r = os.kill(pids[n], signal.SIGINT)  #make sure this is warm not cold shutdown - sigterm seems to be cold
    return


if __name__ == "__main__":
    unique = constants.unique_in_multi_queue
    while 1:
        cpu  = psutil.cpu_percent()
        n_extra = count_queue_workers(unique)
        print(str(n_extra)+' nonpd workers, cpu='+str(cpu))
        if cpu < constants.lower_threshold and n_extra<constants.N_max_workers:
            print('cpu {0} too low, start non-pd worker'.format(cpu))
            start_workers(constants.multi_queue_command,1)
        elif cpu > constants.upper_threshold and n_extra > 0:
            print('cpu {0} too high, kill non-pd worker'.format(cpu))
            kill_worker(unique)

        time.sleep(10)

