#!/usr/bin/env bash
#script to send nvidia status to extremeli
#accessable at http://extremeli.trendi.guru/demo/results/progress_plots/
#crontab line (made using crontab -e) is
#0,10,20,30,40,50 * * * * /usr/lib/python2.7/dist-packages/trendi/calssifier_stuff/auto_nvidia_status.sh
#meaning every 10 minutes, for every hr/day/etc run this script

#get nvidia data
nvidname=$(hostname)_nvidia_output.txt
nvidia-smi > $nvidname
scp $nvidname root@104.155.22.95:/var/www/results/gpu_statii/$nvidname;
