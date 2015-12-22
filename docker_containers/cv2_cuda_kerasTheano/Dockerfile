#FROM ubuntu:14.04
FROM tleyden5iwx/ubuntu-cuda
#increase the available sources for installs
RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list
RUN apt-get update
#RUN apt-get -y upgrade
#RUN apt-key update && apt-get update

RUN apt-get install -y python wget
#RUN apt-get install -y screen
#cmap is  for debugging port forwarding
#RUN apt-get install -y nmap
RUN apt-get install -y cmake   #have below
RUN apt-get install -y unzip

#PYTHON NUMPY
#RUN add-apt-repository ppa:fkrull/deadsnakes
RUN apt-get update
#RUN apt-get install -y python2.7

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
#RUN pip install pymongo
RUN pip install ipython
RUN apt-get install -y python-dev
#RUN apt-get install -y python-numpy
#RUN easy_install numpy
RUN pip install numpy
#RUN pip install python-dateutil
RUN pip install pyparsing
RUN pip install pytz
#RUN pip install matplotlib
RUN apt-get install -y git


################
#numpy scipy
###############
RUN sudo apt-get install -y \
    python-numpy  \
    python-scipy  \
    python-matplotlib   \
    ipython   \
    ipython-notebook    \
    python-pandas     \
    python-sympy     \
    python-nose

###############
#Theano
###############
RUN pip install git+git://github.com/Theano/Theano.git


###############
# Keras
###############
#numpy, scipy donen above

#pyyaml
RUN apt-get install -y libyaml-dev libpython2.7-dev
RUN pip install pyyaml

#HDF5 and h5py (optional, required if you use model saving/loading functions)
RUN apt-get install -y libhdf5-serial-dev
RUN pip install h5py

#Optional but recommended if you use CNNs: cuDNN.
#When using the Theano backend: - Theano - See installation instructions.
#Note: You should use the latest version of Theano, not the PyPI version.
#Install it with:*sudo pip install git+git://github.com/Theano/Theano.git*
#When using the TensorFlow backend: - TensorFlow - See installation instructions.
#To install Keras, cd to the Keras folder and run the install command:​sudo python setup.py install

RUN pip install keras




#these scripts are to check port forwarding with a single command
#RUN  echo 'nmap -p 27017 localhost' > portmongo.sh
#RUN chmod o+x portmongo.sh
#RUN  echo 'nmap -p 6379 localhost' > portredis.sh
#RUN chmod o+x portredis.sh

################################
# Tensorflow
################################
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl


##############################
#OPENCV 3.0 WITH GPU
##############################

MAINTAINER Jeremy Rutman <jeremy@trendiguru.com>

#OPENCV OPENCV OPENCV
#install opencv3.0
RUN apt-get -qq install -y build-essential
RUN apt-get -qq install -y cmake
RUN apt-get -qq install -y pkg-config
RUN apt-get -qq install -y yasm
RUN apt-get -qq install -y libpng12-dev
#RUN apt-get -qq install -y python-dev    #have below
RUN apt-get -qq install -y python-numpy  #have below

#install opengl
#RUN apt-get install -y freeglut3-dev
RUN  mkdir -p /opt/OpenCV
WORKDIR /opt/OpenCV
RUN  wget -O OpenCV3.0.zip https://codeload.github.com/Itseez/opencv/zip/3.0.0
RUN  unzip OpenCV3.0.zip

ENV OPENCV_HOME /opt/OpenCV

#add the nonfree stuff here if necessary
RUN  mkdir -p $OPENCV_HOME/opencv-3.0.0/build
WORKDIR /opt/OpenCV/opencv-3.0.0/build

#for fisherfaces: -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_directory>
#RUN  cmake  .. | tee cmakeout.txt
#below is for gpu-ready cv2
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 .. | tee cmakeout.txt

#RUN  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
# -D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF -D BUILD_opencv_java=OFF \
# -D WITH_IPP=OFF -DOPENCV_EXTRA_MODULES_PATH=/opt/OpenCV/opencv_contrib-3.0.0/modules -D BUILD_NEW_PYTHON_SUPPORT=ON  -D WITH_QT=OFF .. | tee cmakeout.txt

#RUN  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON \
# -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D BUILD_opencv_nonfree=ON ..

RUN  make -j8 | tee makeout.txt
RUN  make install | tee installout.txt

##configure lib search path
RUN  sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN  ldconfig
RUN  echo "OpenCV 3.0.0 is ready to be used"
#RUN printf '# OpenCV\nPKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig\nexport PKG_CONFIG_PATH\n' >> ~/.bashrc
#RUN source ~/.bashrc

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



