__author__ = 'jeremy'
import subprocess, signal
import time
import os
import socket
import argparse

from trendi import constants
import psutil


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
    return n

def start_workers(command,n_workers):

#worker_base_command = '/usr/bin/python /usr/local/bin/rqworker '
#worker_extra_queues={'new_images','find_similar','find_top_n']
#N_expected_pd_workers_extra_queues={47,47,47]

#    command = constants.worker_base_command+queuename
    host = socket.gethostname()
    print('host:'+str(host)+' trying to start '+str(n_workers)+' workers with command '+str(command))
#    if host == 'braini1' or host == 'brain2' or host== 'brain3':
 #   /usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
    for i in range(0,n_workers):
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
    kill_all_workers()
    time.sleep(2)
    for i in range(0,len(constants.worker_commands)):
        n_workers = constants.N_expected_workers[i]
        command = constants.worker_commands[i]
        start_workers(command,n_workers)
        time.sleep(10)

if __name__ == "__main__":
    #scputimes(user=0.3, nice=0.0, system=0.2, idle=99.4, iowait=0.0, irq=0.0, softirq=0.0, steal=0.0, guest=0.0, guest_nice=0.0)
    host = socket.gethostname()
    print('host:'+str(host))
    if host in constants.N_expected_pd_workers_per_server:
        n_expected_workers = constants.N_expected_workers_by_server[host]
    else:
        n_expected_workers = constants.N_default_workers
    unique_strings_to_look_for_in_rq_command.index(queue)
    commands = constants.worker_commands
    unique_string = constants.unique_strings_to_look_for_in_rq_command[i]
    print('i:' + str(i))
    while 1:
        for i in range(0,len(commands)):
            command = commands[i]
            unique_string = constants.unique_strings_to_look_for_in_rq_command[i]
            n_actual_workers = count_queue_workers(unique_string)
            print(str(n_actual_workers)+' workers online in queue '+str(unique_string))
            if n_actual_workers<n_expected_workers:
                start_workers(command,n_expected_workers-n_actual_workers)
        time.sleep(10)

