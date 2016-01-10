__author__ = 'jeremy'
import subprocess, signal
import time
import os
import socket
import psutil

from trendi import constants
import argparse




def kill_pd_workers():
    #full command to start worker is:
 #   cd /home/jeremy/paperdoll/usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
    p = subprocess.Popen(['ps', '-auxw'], stdout=subprocess.PIPE)
    out, err = p.communicate()
#break ps output down into lines and loop on them...:
    string_to_look_for_in_pd_command = constants.string_to_look_for_in_pd_command
    for line in out.splitlines():
        if string_to_look_for_in_pd_command in line:
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
    string_to_look_for_in_pd_command = constants.string_to_look_for_in_pd_command
    p = subprocess.Popen(['ps', '-aux'], stdout=subprocess.PIPE)
    out, err = p.communicate()
#break ps output down into lines and loop on them...:
    for line in out.splitlines():
        #print line
        if string_to_look_for_in_pd_command in line:
            a = line.split()
            print a
            pid = a[1]
            n = n +1
    return n

def start_pd_workers(n=constants.N_expected_pd_workers_per_server):

    host = socket.gethostname()
    print('host:'+str(host)+' trying to start '+str(n)+' workers')
    command = constants.pd_worker_command_braini1
    print('command:'+command)

    if host == 'pp-2':
        print('running on pp-2 so need gcloud command not softlayer')
        command = constants.pd_worker_command
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

def start_workers(command,n_workers):
    host = socket.gethostname()
    print('host:'+str(host)+' trying to start '+str(n_workers)+' workers with command '+str(command))
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

def kill_worker(unique = 'find_similar'):
    p = subprocess.Popen(['ps', '-auxw'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.splitlines():
        if unique in line:
            a = line.split()
            pid = int(a[1])  #maybe on a different unix the output doesnt have owqnder
            print('pid to kill:'+str(pid)+' using unique string:'+str(unique))
#            pid = int(line.split(None, 1)[1])
            r = os.kill(pid, signal.SIGKILL)
            print r

def restart_workers():
    kill_pd_workers()
    time.sleep(2)
    start_pd_workers()

if __name__ == "__main__":
    host = socket.gethostname()
    print('host:'+str(host))
    cpu  = psutil.cpu_percent()

    parser = argparse.ArgumentParser(description='ye olde shepherd')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    # help='an integer for the accumulator')
    parser.add_argument('--N', default=47,
                        help='how many workers')
    queue_order = constants.queue_order
    parser.add_argument('--extra_queues', default=['find_similar','find_top_n'],
                        help='how many pd workers')
    args = parser.parse_args()
    n_expected_workers = int(args.N)
    print('N:' + str(n_expected_workers))
    while 1:
        cpu  = psutil.cpu_percent()
        n_actual_workers = count_pd_workers()
        print(str(n_actual_workers)+' workers online')
        if n_actual_workers<n_expected_workers:
            start_pd_workers(n_expected_workers-n_actual_workers)

        if cpu < constants.lower_threshold:
            print('cpu too low, start non-pd worker')
            start_workers(constants.multi_queue_command,1)
        elif cpu > constants.upper_threshold:
            print('cpu too high, kill non-pd worker')
            kill_worker(constants.unique_in_multi_queue)
        time.sleep(10)