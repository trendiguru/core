__author__ = 'jeremy'
import subprocess, signal
import time
import os
import socket

from trendi import constants


string_in_pd_command = 'rq.tgworker'


def kill_pd_workers():
    #full command to start worker is:
 #   cd /home/jeremy/paperdoll/usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
    p = subprocess.Popen(['ps', '-auxw'], stdout=subprocess.PIPE)
    out, err = p.communicate()
#break ps output down into lines and loop on them...:
    for line in out.splitlines():
        if string_in_pd_command in line:
            a = line.split()
            pid = int(a[1])  #maybe on a different unix the output doesnt have owqnder
            print('pid to kill:'+str(pid))
#            pid = int(line.split(None, 1)[1])
            r = os.kill(pid, signal.SIGKILL)
            print r

def count_pd_workers():
    n = 0
    #full command to start worker is:
 #   /usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
#    command = 'ps -auxw'
 #   p = subprocess.Popen(command, shell=True,stdout=subprocess.PIPE).stdout.read()
    p = subprocess.Popen(['ps', '-auxw'], stdout=subprocess.PIPE)
    out, err = p.communicate()
#break ps output down into lines and loop on them...:
    for line in out.splitlines():
#        print line
        if string_in_pd_command in line:
            a = line.split()
            print a
            pid = a[1]
            n = n +1
    return n

def start_pd_workers(n=constants.N_expected_pd_workers_per_server):

    command = constants.pd_worker_command
    host = socket.gethostname()
    print('host:'+str(host)+' trying to start '+str(n)+' workers')
    if host == 'braini1':
        command = constants.pd_worker_command_braini1
 #   /usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
    for i in range(0,n):
  #      command = 'cd /home/jeremy/paperdoll3/paperdoll-v1.0/'
        print('command:'+command)
#        p = subprocess.Popen(command, shell=True,stdout=subprocess.PIPE).stdout.read()
        p = subprocess.Popen(command, shell=True,stdout=subprocess.PIPE)
        print(p)
        #p = subprocess.Popen(command, stdout=subprocess.PIPE)
        #command =  '/usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd'
        #print('command:'+command)
        #p = subprocess.Popen(command, stdout=subprocess.PIPE)

def restart_workers():
    kill_pd_workers()
    time.sleep(2)
    start_pd_workers()

if __name__ == "__main__":
    host = socket.gethostname()
    print('host:'+str(host))
    if host == 'braini1':
        n_workers = constants.N_expected_pd_workers_per_server_braini1
    else:
        n_workers = constants.N_expected_pd_workers_per_server
    while 1:
        n = count_pd_workers()
        print(str(n)+' workers online')
        if n<n_workers:
            start_pd_workers(n_workers-n)
        time.sleep(10)