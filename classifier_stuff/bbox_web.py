__author__ = 'jeremy'
import os
from subprocess import call

def del(image_number):
    command = 'ssh root@104.155.22.95 mv /home/jeremy/trainjr.txt /home/jeremy/MOVEDITBABY.txt'
    call(command)
    print('finished command')
    with open('out.txt','a') as f:
        f.write('did '+command)
        f.close()