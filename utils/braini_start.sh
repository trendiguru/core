#!/bin/bash
screen -dRR ssh
lsof -ti:6379 | xargs kill -9
lsof -ti:27017 | xargs kill -9
ssh -f -N -L 6379:redis1-redis-1-vm:6379 root@extremeli.trendi.guru
ssh -f -N -L 27017:mongodb1-instance-1:27017 root@extremeli.trendi.guru

#how to get out of screen without using ctrl-A d ?
screen -dRR shepherd
cd /home/pd_user/trendi/paperdoll
python doll_shepherd.py --N=47

screen -dRR major
cd /home/pd_user/trendi/paperdoll
python major_shepherd.py