#!/usr/bin/env bash
#link protos in /home/jeremy/caffenets to sources in core (/usr/lib/python2.7/dist-packages/trendi
#only loose ends are a. rest of protos and b. caffemodels, which need to be in /home/jeremy/caffenets
#caffemodels are big and hence not included in trendi

for f in *;
    do echo $f ;
    ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/protos/multilabel/resnet/$f /home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/$f ;
done

#get jrlayers into caffe path
ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/jrlayers.py /opt/caffe/python/jrlayers.py

# get images db
# rsync -r root@37.58.101.173:/home/jeremy/image_dbs/tamara_berg /home/jeremy/image_dbs/
# get caffenets
# rsync -r root@37.58.101.173:/home/jeremy/caffenets /home/jeremy/