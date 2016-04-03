__author__ = 'jeremy'
import os
from subprocess import call

def do_delete(image_number):
    braini2_ip = '37.58.101.173'
    print('finished command')
    command = 'ssh root@'+braini2_ip+' mv /home/jeremy/trainjr.txt /home/jeremy/MOVEDITBABY.txt'
    os.system(command)
    print('finished command')
    with open('out.txt','a') as f:
        f.write('did '+command+'\n')
        f.close()