#!/usr/bin/env bash
#script to produce plots from recent caffe logfiles , send them to extremeli
# accessable at http://extremeli.trendi.guru/demo/results/progress_plots/

#produce the plots from any logfile updated in last 100 minutes
counter = 0
logfiles="$(find /tmp caffe* -mmin -100|grep -v jpg|grep -v caffe.INFO)"
echo $logfiles
for log in $logfiles;
   do echo $log;
   python /home/jeremy/core/classifier_stuff/caffe_nns/progress_plot.py --log True $log;
done

#send any .jpg  updated in last 100 minutes
jpgfiles="$(find /tmp caffe* -mmin -100|grep jpg)"
echo $jpgfiles
for jpg in $jpgfiles;
   do echo $jpg;
   counter=$((counter+1))
   newname="$counter.jpg"
   echo $newname
   scp $jpg root@104.155.22.95:/var/www/results/progress_plots/$newname;
#   rsync jpg root@37.58.64.220:/var/www/results/progress_plots;
done




