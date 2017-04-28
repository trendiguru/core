#!/usr/bin/env bash

wget http://host.robots.ox.ac.uk/pascal/VOC/download/voc2005_1.tar.gz
wget http://host.robots.ox.ac.uk/pascal/VOC/download/voc2005_2.tar.gz

wget http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_trainval.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_test.tar

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar
#wget http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2008test.tar
#the test files need permission from the server

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar
#wget http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2009test.tar

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
#wget http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
#wget http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar

tar -xf voc2005_1.tar.gz
tar -xf voc2005_2.tar.gz

tar -xf voc2006_trainval.tar
tar -xf voc2006_test.tar

tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar

tar -xf VOCtrainval_14-Jul-2008.tar
tar -xf VOC2008test.tar

tar -xf VOCtrainval_11-May-2009.tar
tar -xf VOC2009test.tar

tar -xf VOCtrainval_03-May-2010.tar
tar -xf VOC2010test.tar

tar -xf VOCtrainval_11-May-2012.tar
tar -xf VOC2012test.tar

#http://host.robots.ox.ac.uk/pascal/VOC/download/caltech.tar.gz
#http://host.robots.ox.ac.uk/pascal/VOC/download/mit-000.tar.gz
#http://host.robots.ox.ac.uk/pascal/VOC/download/mit-001.tar.gz
#http://host.robots.ox.ac.uk/pascal/VOC/download/tug.tar.gz
#http://host.robots.ox.ac.uk/pascal/VOC/download/uiuc.tar.gz
#http://host.robots.ox.ac.uk/pascal/VOC/download/tud.tar.gz