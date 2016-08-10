#!/usr/bin/env bash
#script to produce plots from recent caffe logfiles , send them to extremeli
# accessable at http://extremeli.trendi.guru/demo/results/progress_plots/
#currently this script runs on braini2 through crontab
#crontab line (made using crontab -e) is
#0,10,20,30,40,50 * * * * /root/auto_progress_plots.sh
#meaning every 10 minutes, for every hr/day/etc run this script

#produce the plots from any logfile updated in last 100 minutes
counter = 0
logfiles="$(find /tmp caffe* -mmin -100|grep -v jpg|grep -v caffe.INFO|grep caffe.)"
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

#do multilabel accuracy/precision/recall tests
snapshot_root="/home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/"
snapshot_dir[1]="snapshot_50_B"
snapshot_dir[2]="snapshot_50_sgd"
snapshot_dir[3]="snapshot101"
snapshot_dir[4]="snapshot101_sgd"
snapshot_dir[5]="snapshot_152"
protos[1]="ResNet-50-test.prototxt"
protos[2]="ResNet-50-test.prototxt"
protos[3]="ResNet-101-test.prototxt"
protos[4]="ResNet-101-test.prototxt"
protos[5]="ResNet-152-test.prototxt"


logfile[1]="$(ls -tr ${snapshot_dir[1]}/*caffemodel |tail -1)"
logfile[2]="$(ls -tr ${snapshot_dir[2]}/*caffemodel |tail -1)"
logfile[3]="$(ls -tr ${snapshot_dir[3]}/*caffemodel |tail -1)"
logfile[4]="$(ls -tr ${snapshot_dir[4]}/*caffemodel |tail -1)"
logfile[5]="$(ls -tr ${snapshot_dir[5]}/*caffemodel |tail -1)"

#logfile5="$(ls -tr /home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/$snapshot_dir5/*caffemodel |tail -1)"
echo ${logfile[1]}

echo $logfile
counter=0
for i in 1 2 3 4 5;

   do echo ${logfile[$i]};
   caffemod=${logfile[i]}
   proto=${protos[i]};
#   echo "proto"
#   echo $proto;
   com="python /home/jeremy/core/classifier_stuff/caffe_nns/multilabel_accuracy.py --caffemodel "$caffemod ;
   com=$com" --testproto ";
   com=$com$proto;
#   echo "cpm"
   echo $com;
   $com;
done

#send any .jpg  updated in last 100 minutes
newfiles="$(find /tmp caffe* -mmin -100|grep jpg)"
echo $newfiles
for f in $newfiles;
   do echo $f;
   let "counter=counter+1"
   newname="$counter.jpg"
   echo $newname
   scp $f root@104.155.22.95:/var/www/results/progress_plots/$newname;
#   rsync jpg root@37.58.64.220:/var/www/results/progress_plots;
done


logfile[1]="net_output.txt"
#logfile5="$(ls -tr /home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/$snapshot_dir5/*caffemodel |tail -1)"
for i in 1;
   do echo ${logfile[$i]};
   logf=${logfile[i]}
   com = "python /home/jeremy/core/classifier_stuff/caffe_nns/progress_plot.py " $logf " --type 1"
   echo $com;
   $com;
done

echo $counter
let "counter=counter+1"
echo $counter
newname="$counter.jpg"
echo $newname
scp  /home/jeremy/caffenets/pixlevel/voc-fcn8s/voc8.15/net_output.txt.jpg root@104.155.22.95:/var/www/results/progress_plots/$newname;
