#!/bin/sh
#put this file (or better yet a symlink thereto) at /etc/init.d/portforward_initscript.sh
#it is supposed to get the port forwarding for brainis working
ssh -f -N -L 27017:mongodb1-instance-1:27017 root@extremeli.trendi.guru
ssh -f -N -L 6379:redis1-redis-1-vm:6379 root@extremeli.trendi.guru
python /home/pd_user/trendi/paperdoll/doll_shepherd.py