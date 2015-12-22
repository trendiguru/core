__author__ = 'jeremy'
import subprocess, signal


def kill_pd_workers():
    string_in_pd_command = 'rq.tgworker'
    #full command to start worker is:
 #   /usr/bin/python /usr/local/bin /rqworker -w rq.tgworker.TgWorker -u redis://redis1-redis-1-vm:6379 pd
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
#break ps output down into lines and loop on them...:
    for line in out.splitlines():
        if string_in_pd_command in line:
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)

def start_pd_workers():

if __name__ == "__main__"
