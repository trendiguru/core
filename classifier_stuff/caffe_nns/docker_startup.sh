#!/usr/bin/env bash
#run this when starting a container so that the python layers from shelhamer get linked
#not needed if you just train with 'caffe train ...' or the like
ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/score.py /opt/caffe/python/score.py
ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/surgery.py /opt/caffe/python/surgery.py
ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/jrlayers.py /opt/caffe/python/jrlayers.py
#cp /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/solve.py

#these are unecessary with new build of container (setproctitle put in to requirements.txt)
pip install setproctitle
apt-get install tmux
