#!/bin/bash

cd /root
git clone https://github.com/BVLC/caffe.git
cd caffe

# Install python dependencies
cat python/requirements.txt | xargs -n1 pip install && \

#prereqs for cuda
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev

sudo apt-get update
sudo apt-get install -y  bc
sudo apt-get install -y  cmake
sudo apt-get install -y  curl
sudo apt-get install -y  gcc-4.6
sudo apt-get install -y  g++-4.6
sudo apt-get install -y  gcc-4.6-multilib
sudo apt-get install -y  g++-4.6-multilib
sudo apt-get install -y  gfortran
sudo apt-get install -y  git
sudo apt-get install -y  libprotobuf-dev
sudo apt-get install -y  libleveldb-dev
sudo apt-get install -y  libsnappy-dev
sudo apt-get install -y  libopencv-dev
sudo apt-get install -y  libboost-all-dev
sudo apt-get install -y  libhdf5-serial-dev
sudo apt-get install -y  liblmdb-dev
sudo apt-get install -y  libjpeg62
sudo apt-get install -y  libfreeimage-dev
sudo apt-get install -y  libatlas-base-dev
sudo apt-get install -y  pkgconf
sudo apt-get install -y  protobuf-compiler
sudo apt-get install -y  python-dev
sudo apt-get install -y  python-pip
sudo apt-get install -y  unzip
sudo apt-get install -y  wget

#sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
#sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
#sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libboost-all-dev

# glog
wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
tar zxvf glog-0.3.3.tar.gz
cd glog-0.3.3
./configure
make && make install

# gflags
wget https://github.com/schuhschuh/gflags/archive/master.zip
unzip master.zip
cd gflags-master
mkdir build && cd build
export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
make && make install

# lmdb
git clone git://gitorious.org/mdb/mdb.git
cd mdb/libraries/liblmdb
make && make install

# Make and move into build directory
cp Makefile.config.example Makefile.config
sed -i 's/# USE_CUDNN := 1/USE_CUDNN := 1/' Makefile.config
sed -i 's/# OPENCV_VERSION := 3/OPENCV_VERSION := 3/' Makefile.config
sed -i 's/CUDA_DIR := \/usr\/local\/cuda/CUDA_DIR := \/usr\/local\/cuda-7.5/' Makefile.config
mkdir build && cd build

# CMake
#cmake ..

# Make
#make -j"$(nproc)" all
#make -j45 all

#make pycaffe
