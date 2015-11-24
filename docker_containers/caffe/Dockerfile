FROM ubuntu:14.04


###################
#CAFFE
###################
#caffe reqs:
#anaconda

 #libgflags.a, libgflags_nothreads.a,


#RUN sudo apt-get install -y libgflags-dev


ENV PYTHONPATH /opt/caffe/python
# Add caffe binaries to path
ENV PATH $PATH:/opt/caffe/.build_release/tools

# Get dependencies
RUN apt-get update && apt-get install -y \
  bc \
  cmake \
  curl \
  gcc-4.6 \
  g++-4.6 \
  gcc-4.6-multilib \
  g++-4.6-multilib \
  gfortran \
  git \
  libprotobuf-dev \
  libleveldb-dev \
  libsnappy-dev \
  libopencv-dev \
  libboost-all-dev \
  libhdf5-serial-dev \
  liblmdb-dev \
  libjpeg62 \
  libfreeimage-dev \
  libatlas-base-dev \
  pkgconf \
  protobuf-compiler \
  python-dev \
  python-pip \
  unzip \
  wget

# Use gcc 4.6
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-4.6 30 && \
  update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-4.6 30 && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 30 && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.6 30

# Glog
RUN cd /opt && wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz && \
  tar zxvf glog-0.3.3.tar.gz && \
  cd /opt/glog-0.3.3 && \
  ./configure && \
  make && \
  make install

# Workaround for error loading libglog:
#   error while loading shared libraries: libglog.so.0: cannot open shared object file
# The system already has /usr/local/lib listed in /etc/ld.so.conf.d/libc.conf, so
# running `ldconfig` fixes the problem (which is simpler than using $LD_LIBRARY_PATH)
# TODO: looks like this needs to be run _every_ time a new docker instance is run,
#       so maybe LD_LIBRARY_PATh is a better approach (or add call to ldconfig in ~/.bashrc)
RUN ldconfig

# Gflags
WORKDIR /opt
RUN  wget https://github.com/schuhschuh/gflags/archive/master.zip
RUN  unzip master.zip
WORKDIR gflags-master
RUN  pwd>cwd1.txt
RUN  mkdir build
WORKDIR build
RUN  pwd>cwd2.txt
RUN  export CXXFLAGS="-fPIC"
RUN  cmake ..
RUN  make VERBOSE=1
RUN  make
RUN  make install


# Clone the Caffe repo
WORKDIR /opt
RUN echo $(pwd)
RUN git clone https://github.com/BVLC/caffe.git

# Build Caffe core
WORKDIR /opt/caffe
RUN   cp Makefile.config.example Makefile.config
RUN   echo "CPU_ONLY := 1" >> Makefile.config
RUN  echo "CXX := /usr/bin/g++-4.6" >> Makefile.config
RUN  sed -i 's/CXX :=/CXX ?=/' Makefile
RUN  make all

# Add ld-so.conf so it can find libcaffe.so
ADD caffe-ld-so.conf /etc/ld.so.conf.d/

# Run ldconfig again (not sure if needed)
RUN ldconfig

# Install python deps
RUN cd /opt/caffe && \
  (pip install -r python/requirements.txt; easy_install numpy; pip install -r python/requirements.txt) && \
  easy_install pillow

#apparently not necessary after numpy install above
# Numpy include path hack - github.com/BVLC/caffe/wiki/Setting-up-Caffe-on-Ubuntu-14.04
#RUN NUMPY_EGG=`ls /usr/local/lib/python2.7/dist-packages | grep -i numpy` && \
#  ln -s /usr/local/lib/python2.7/dist-packages/$NUMPY_EGG/numpy/core/include/numpy /usr/include/python2.7/numpy

# Build Caffe python bindings
WORKDIR /opt/caffe
RUN make pycaffe


# Make + run tests
WORKDIR /opt/caffe
RUN make test && make runtest

###### get alexnet
#make all test#
#make runtest
#alexnet - get model

WORKDIR /opt/caffe

RUN ./scripts/download_model_binary.py ./models/bvlc_reference_caffenet
RUN ./data/ilsvrc12/get_ilsvrc_aux.sh

RUN ./build/examples/cpp_classification/classification.bin ./models/bvlc_reference_caffenet/deploy.prototxt ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel ./data/ilsvrc12/imagenet_mean.binaryproto ./data/ilsvrc12/synset_words.txt ./examples/images/cat.jpg


################################
# git clone the tg repo , run this last since it requires nonsudo user
################################

## TG GIT CLONE ##
## TG GIT CLONE ##
## TG GIT CLONE ##

RUN apt-get update
RUN apt-get install -y ca-certificates
#RUN apt-get install -y git-core
RUN apt-get install -y --force-yes git
RUN apt-get install -y ssh

ENV HOME /home/developer
ADD ssh/ /home/developer/.ssh/

ADD ssh/ssh_config /etc/ssh/ssh_config

# Fixes permission if needed
#RUN chmod 600 /.ssh/*
RUN chmod 600 /home/developer/.ssh/*
RUN chown developer:developer /home/developer/
ADD ssh/first_aws.pem /home/developer/.ssh/first_aws.pem
RUN chown developer:developer /home/developer/.ssh/*

RUN ssh-keyscan  bitbucket.com > /home/developer/.ssh/known_hosts

RUN mkdir /home/developer/tg
RUN chown developer:developer /home/developer/tg
RUN mkdir /home/developer/tg/Dev
RUN chown developer:developer /home/developer/tg/Dev
WORKDIR /home/developer/tg/Dev

RUN apt-get -y --force-yes  install nano

USER developer
RUN whoami
RUN git clone --verbose git@bitbucket.org:trendiGuru/core.git

#mongobd port forward
CMD ssh -i /.ssh/first_aws.pem -L 27017:localhost:27017 ubuntu@extremeli.trendi.guru && /bin/bash

RUN rqworker -u redis://redis1-redis-1-vm:6379 caffe_general &
#RUN rqworker gender &
