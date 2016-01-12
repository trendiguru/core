
import subprocess

    def info():
        command = 'rqinfo -u redis://redis1-redis-1-vm:6379''
        retval = subprocess.call(command, shell=True)  # create the svg
        print retval
