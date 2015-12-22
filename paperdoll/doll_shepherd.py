__author__ = 'jeremy'
import subprocess, signal
import time

import constants

string_in_pd_command = 'rq.tgworker'


def kill_pd_workers():
    #full command to start worker is:
 #   cd /home/jeremy/paperdoll/usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
#break ps output down into lines and loop on them...:
    for line in out.splitlines():
        if string_in_pd_command in line:
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)

def count_pd_workers():
    n = 0
    #full command to start worker is:
 #   /usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
#break ps output down into lines and loop on them...:
    for line in out.splitlines():
        if string_in_pd_command in line:
            pid = int(line.split(None, 1)[0])
            n = n +1
    return n

def start_pd_workers(n=constants.N_expected_pd_workers_per_server):
 #   /usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
    for i in range(0,n)
        print('attempting to start worker')
        command = constants.pd_worker_command
        p = subprocess.Popen([command], stdout=subprocess.PIPE)

def restart_workers():
    kill_pd_workers()
    time.sleep(2)
    start_pd_workers()

if __name__ == "__main__"
    while 1:
        n = count_pd_workers()
        print(str(n)+' workers online')
        if n<constants.N_expected_pd_workers_per_server
            start_pd_workers(constants.N_expected_pd_workers_per_server-n)
        time.sleep(10)