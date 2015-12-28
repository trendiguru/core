#RUN echo deb http://archive.ubuntu.com/ubuntu precise universe multiverse >> /etc/apt/sources.list; \
sudo   apt-get update -qq && apt-get install -y --force-yes \
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
    python-dev \
    python-numpy \
    default-jdk \
    ant \
    wget \
    unzip; \
    apt-get clean


RUN apt-get update
#RUN apt-get -y upgrade
#RUN apt-key update && apt-get update

RUN apt-get install -y python wget
RUN apt-get install -y screen
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
RUN pip install pymongo
RUN pip install ipython
RUN apt-get install -y python-dev
#RUN apt-get install -y python-numpy
#RUN easy_install numpy
RUN pip install numpy
#RUN pip install python-dateutil
RUN pip install pyparsing
RUN pip install pytz
#RUN pip install matplotlib



YASM_VERSION=1.3.0
OPENCV_VERSION=3.0
cd /usr/local/src
git clone --depth 1 https://github.com/l-smash/l-smash
git clone --depth 1 git://git.videolan.org/x264.git
git clone https://bitbucket.org/multicoreware/x265
git clone --depth 1 git://source.ffmpeg.org/ffmpeg
git clone https://github.com/Itseez/opencv.git
git clone --depth 1 git://github.com/mstorsjo/fdk-aac.git
git clone --depth 1 https://chromium.googlesource.com/webm/libvpx
git clone --depth 1 git://git.opus-codec.org/opus.git
git clone --depth 1 https://github.com/mulx/aacgain.git
curl -Os http://www.tortall.net/projects/yasm/releases/yasm-${YASM_VERSION}.tar.gz
tar xzvf yasm-${YASM_VERSION}.tar.gz
# Build YASM # =================================
cd /usr/local/src/yasm-${YASM_VERSION}
./configure
make -j 48
make install
# ================================= # Build L-SMASH # =================================
cd /usr/local/src/l-smash
./configure
make -j 4
make install
# ================================= # Build libx264 # =================================
cd /usr/local/src/x264
./configure --enable-static
make -j 4
make install
# ================================= # Build libx265 # =================================
cd /usr/local/src/x265/build/linux
cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr ../../source
make -j 4
make install
# ================================= # Build libfdk-aac # =================================
cd /usr/local/src/fdk-aac
autoreconf -fiv
./configure --disable-shared
make -j 4
make install
# ================================= # Build libvpx # =================================
cd /usr/local/src/libvpx
./configure --disable-examples
make -j 4
make install
# ================================= # Build libopus # =================================
cd /usr/local/src/opus
./autogen.sh
./configure --disable-shared
make -j 4
make install
# ================================= # Build OpenCV 3.0.0 # =================================
cd /usr/local/src
apt-get update -qq 
#apt-get install -y --force-yes libopencv-dev
#git clone https://github.com/Itseez/opencv.git
#wget https://github.com/Itseez/opencv/archive/3.0.0.zip
#unzip 3.0.0.zip
#mkdir -p opencv/release
#echo ls

git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.0.0
git status
git branch
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..
make -j48
sudo make install
sudo ldconfig

export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python2.7/site-packages
sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'

#cd "matlabroot/extern/engines/python"
#python setup.py install

#VNC
sudo apt-get update
sudo apt-get install xfce4 xfce4-goodies tightvncserver
adduser vnc
sudo su vnc
vncserver