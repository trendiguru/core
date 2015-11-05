FROM ubuntu:14.04

#increase the available sources for installs
#RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-key update

RUN apt-get install -y wget
RUN apt-get install -y screen
#cmap is  for debugging port forwarding
RUN apt-get install -y nmap
RUN apt-get install -y cmake
RUN apt-get install -y unzip

#PYTHON NUMPY
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install pymongo
RUN pip install ipython
RUN apt-get install -y python-dev
#RUN apt-get install -y python-numpy
#RUN easy_install numpy
RUN pip install numpy
RUN pip install python-dateutil
RUN pip install pyparsing
RUN pip install pytz
#RUN pip install matplotlib


#these scripts are to check port forwarding with a single command
RUN  echo 'nmap -p 27017 localhost' > portmongo.sh
RUN chmod o+x portmongo.sh
RUN  echo 'nmap -p 6379 localhost' > portredis.sh
RUN chmod o+x portredis.sh

##############################
#OPENCV
##############################

#increase the available sources for installs
#RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-key update

#not sure if build-essential is essential
RUN apt-get install -y build-essential
RUN apt-get install -y pkg-config
RUN apt-get install -y yasm
RUN apt-get install -y wget
RUN apt-get install -y libpng-dev
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y freetype*
RUN apt-get install -y python-tk
RUN apt-get install -y screen
#cmap is  for debugging port forwarding
RUN apt-get install -y nmap
#RUN apt-get install -y checkinstall
RUN apt-get install -y cmake
RUN apt-get install -y unzip

#PYTHON NUMPY
#not sure why the below is used
#RUN checkinstall cmake pkg-config yasm unzip wget
RUN apt-get install -y python-dev
RUN apt-get install -y python-numpy
RUN apt-get autoclean
RUN rm -rf /var/lib/apt/lists/*
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install pymongo
RUN pip install ipython
RUN pip install numpy
RUN pip install python-dateutil
RUN pip install pyparsing
RUN pip install pytz
RUN pip install matplotlib

RUN apt-get -qq install -y build-essential
RUN apt-get -qq install -y cmake
RUN apt-get -qq install -y pkg-config
RUN apt-get -qq install -y yasm
RUN apt-get -qq install -y libpng12-dev
RUN apt-get -qq install -y python-dev
RUN apt-get -qq install -y python-numpy

RUN  mkdir -p /opt/OpenCV
RUN  cd /opt/OpenCV
RUN pwd
WORKDIR /opt/OpenCV
RUN  wget -O OpenCV3.0.zip https://codeload.github.com/Itseez/opencv/zip/3.0.0
RUN  unzip OpenCV3.0.zip
#compiling contrib opencv stuff requires dl of a further repo at https://github.com/Itseez/opencv_contrib.git
#RUN  wget -O opencv_contrib.zip https://codeload.github.com/Itseez/opencv_contrib/zip/3.0.0
#RUN  unzip opencv_contrib.zip

ENV OPENCV_HOME /opt/OpenCV
#add the nonfree stuff here if necessary
RUN  mkdir -p $OPENCV_HOME/opencv-3.0.0/build
WORKDIR /opt/OpenCV/opencv-3.0.0/build

#for fisherfaces: -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_directory>
RUN  cmake  .. | tee cmakeout.txt

#RUN  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
# -D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF -D BUILD_opencv_java=OFF \
# -D WITH_IPP=OFF -DOPENCV_EXTRA_MODULES_PATH=/opt/OpenCV/opencv_contrib-3.0.0/modules -D BUILD_NEW_PYTHON_SUPPORT=ON  -D WITH_QT=OFF .. | tee cmakeout.txt

RUN  make -j8 | tee makeout.txt
RUN  make install | tee installout.txt

RUN  sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN  ldconfig
RUN  echo "OpenCV 3.0.0 is ready to be used (maybe, check python->import cv2"



################
#unknwon maybe remove
###################
RUN export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH

RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer


#####################
# redis and rq
#####################
#RUN pip install redis
RUN pip install rq

#####################
# ssh port forwarding
#####################
#turn off asking 'are you sure' on first ssh connect
#this is supposed to be taken care of by the keyscan but isn't , perhaps because this
# is not our own repo running on a port like in the example
#ADD ssh/ssh_config /etc/ssh/ssh_config

#mongobd port forward
#CMD ssh -i /.ssh/first_aws.pem -L 27017:localhost:27017 ubuntu@extremeli.trendi.guru && /bin/bash

#these scripts are to check port forwarding with a single command
RUN  echo 'nmap -p 27017 localhost' > portmongo.sh
RUN chmod o+x portmongo.sh
RUN  echo 'nmap -p 6379 localhost' > portredis.sh
RUN chmod o+x portredis.sh

################
#numpy scipy
###############
RUN sudo apt-get install -y python-numpy
RUN sudo apt-get install -y python-scipy
RUN sudo apt-get install -y python-matplotlib
RUN sudo apt-get install -y ipython
RUN sudo apt-get install -y ipython-notebook
RUN sudo apt-get install -y python-pandas
RUN sudo apt-get install -y python-sympy
#    python-nose


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



# Clone the Caffe repo
RUN cd /opt && git clone https://github.com/BVLC/caffe.git


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
RUN cd /opt
RUN  wget https://github.com/schuhschuh/gflags/archive/master.zip
RUN  unzip master.zip
RUN  cd /opt/gflags-master
RUN  mkdir build
RUN  cd /opt/gflags-master/build
RUN  export CXXFLAGS="-fPIC"
RUN  cmake ..
RUN  make VERBOSE=1
RUN  make
RUN  make install

# Build Caffe core
RUN cd /opt/caffe
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
RUN cd /opt/caffe && make pycaffe


# Make + run tests
RUN cd /opt/caffe && make test && make runtest

###### get alexnet 0
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
