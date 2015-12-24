FROM ubuntu:14.04
#FROM tleyden5iwx/ubuntu-cuda
#FROM ubuntu:trusty
# https://github.com/ampervue/docker-python27-opencv
#MAINTAINER David Karchmer <dkarchmer@gmail.com>
######################################## # # Image based on Ubuntu:trusty # #   with Python 2.7 #   and OpenCV 3 (built) #
#plus a bunch of build essencials ####################################### #
#Set Locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN apt-get -qq remove ffmpeg


##############################
#OPENCV 3.0 WITH GPU
##############################

RUN echo deb http://archive.ubuntu.com/ubuntu precise universe multiverse >> /etc/apt/sources.list; \
    apt-get update -qq && apt-get install -y --force-yes \
    curl \
    git \
    g++ \
    autoconf \
    automake \
    mercurial \
    libopencv-dev \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    libtiff4-dev \
    libpng-dev \
    libjpeg-dev \
    libjasper-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine-dev \
    libgstreamer0.10-dev \
    libgstreamer-plugins-base0.10-dev \
    libv4l-dev \
    libtbb-dev \
    libgtk2.0-dev \
    libfaac-dev \
    libmp3lame-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libtool \
    v4l-utils \
    python2.7 \
    python2.7-dev \
    python-numpy \
    default-jdk \
    ant \
    wget \
    unzip; \
    apt-get clean
ENV YASM_VERSION 1.3.0
ENV OPENCV_VERSION 2.4.10
WORKDIR /usr/local/src
RUN git clone --depth 1 https://github.com/l-smash/l-smash
RUN git clone --depth 1 git://git.videolan.org/x264.git
RUN hg clone https://bitbucket.org/multicoreware/x265
RUN git clone --depth 1 git://source.ffmpeg.org/ffmpeg
RUN git clone https://github.com/Itseez/opencv.git
RUN git clone --depth 1 git://github.com/mstorsjo/fdk-aac.git
RUN git clone --depth 1 https://chromium.googlesource.com/webm/libvpx
RUN git clone --depth 1 git://git.opus-codec.org/opus.git
RUN git clone --depth 1 https://github.com/mulx/aacgain.git
RUN curl -Os http://www.tortall.net/projects/yasm/releases/yasm-${YASM_VERSION}.tar.gz
RUN tar xzvf yasm-${YASM_VERSION}.tar.gz
# Build YASM # =================================
WORKDIR /usr/local/src/yasm-${YASM_VERSION}
RUN ./configure
RUN make -j 4
RUN make install
# ================================= # Build L-SMASH # =================================
WORKDIR /usr/local/src/l-smash
RUN ./configure
RUN make -j 4
RUN make install
# ================================= # Build libx264 # =================================
WORKDIR /usr/local/src/x264
RUN ./configure --enable-static
RUN make -j 4
RUN make install
# ================================= # Build libx265 # =================================
WORKDIR /usr/local/src/x265/build/linux
RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr ../../source
RUN make -j 4
RUN make install
# ================================= # Build libfdk-aac # =================================
WORKDIR /usr/local/src/fdk-aac
RUN autoreconf -fiv
RUN ./configure --disable-shared
RUN make -j 4
RUN make install
# ================================= # Build libvpx # =================================
WORKDIR /usr/local/src/libvpx
RUN ./configure --disable-examples
RUN make -j 4
RUN make install
# ================================= # Build libopus # =================================
WORKDIR /usr/local/src/opus
RUN ./autogen.sh
RUN ./configure --disable-shared
RUN make -j 4
RUN make install
# ================================= # Build OpenCV 3.x # =================================
RUN apt-get update -qq && apt-get install -y --force-yes libopencv-dev
WORKDIR /usr/local/src
RUN mkdir -p opencv/release
WORKDIR /usr/local/src/opencv/release
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_TBB=ON \
          -D BUILD_PYTHON_SUPPORT=ON \
          -D WITH_V4L=ON \
          ..
RUN make -j4
RUN make install
RUN sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN ldconfig
# ================================= # Build ffmpeg. # =================================
RUN apt-get update -qq && apt-get install -y --force-yes \
    libass-dev
WORKDIR /usr/local/src/ffmpeg
RUN ./configure --extra-libs="-ldl" \
            --enable-gpl \
            --enable-libass \
            --enable-libfdk-aac \
            --enable-libfontconfig \
            --enable-libfreetype \
            --enable-libfribidi \
            --enable-libmp3lame \
            --enable-libopus \
            --enable-libtheora \
            --enable-libvorbis \
            --enable-libvpx \
            --enable-libx264 \
            --enable-libx265 \
            --enable-nonfree
RUN make -j 4
RUN make install
# ================================= # Remove all tmpfile # =================================
WORKDIR /usr/local/
#RUN rm -rf /usr/local/src
# ================================= # Install pip # =================================
#RUN curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | python



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
#RUN pip install numpy
#RUN pip install python-dateutil
RUN pip install pyparsing
RUN pip install pytz
#RUN pip install matplotlib
RUN apt-get install -y git
RUN pip install -e git+https://github.com/garydoranjr/pyemd.git#egg=pyemd # python package for earth removal calculation

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



#####################
# redis and rq
#####################
#RUN pip install redis
RUN pip install rq


RUN apt-get -y --force-yes  install nano

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

ADD ssh/ /home/root/.ssh/

ADD ssh/ssh_config /etc/ssh/ssh_config

# Fixes permission if needed
#RUN chmod 600 /.ssh/*
RUN chmod 600 /home/root/.ssh/*
ADD ssh/first_aws.pem /home/root/.ssh/first_aws.pem

RUN ssh-keyscan  bitbucket.com > /home/root/.ssh/known_hosts
RUN mkdir /home/root/Dev
WORKDIR /home/root/Dev
#RUN git clone --verbose git@bitbucket.org:trendiGuru/core.git






