#!/usr/bin/env bash
#script to produce plots from recent caffe logfiles , send them to extremeli
# accessable at http://extremeli.trendi.guru/demo/results/progress_plots/
#currently this script runs on braini2 through crontab
#crontab line (made using crontab -e) is
#0,10,20,30,40,50 * * * * /root/auto_progress_plots.sh
#meaning every 10 minutes, for every hr/day/etc run this script

#produce the plots from any logfile updated in last 100 minutes
counter=0
#find caffe logfiles from last 100 minutes
logfiles="$(find /tmp caffe* -mmin -100|grep -v jpg|grep -v caffe.INFO|grep caffe.|grep -v FATAL |grep -v WARNING|grep -v ERROR)"
log_command="/usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/progress_plot.py"
echo $logfiles
for log in $logfiles;
   do echo $log;
   python $log_command --log True $log;
done

#send any .jpg  updated in last 100 minutes
host=$(hostname)
jpgfiles="$(find /tmp caffe* -mmin -100|grep jpg)"
echo $jpgfiles
for jpg in $jpgfiles;
   do echo $jpg;
   counter=$((counter+1))
   newname="$host-$counter.jpg"
   echo $newname
   scp $jpg root@104.155.22.95:/var/www/results/progress_plots/$newname;
#   rsync jpg root@37.58.64.220:/var/www/results/progress_plots;
done
