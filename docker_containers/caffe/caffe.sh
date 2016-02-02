#!/bin/bash

cd /root
git clone https://github.com/BVLC/caffe.git
cd caffe

# Install python dependencies
cat python/requirements.txt | xargs -n1 pip install && \


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

# Make and move into build directory
mkdir build && cd build && \

# CMake
cmake .. && \

# Make
#make -j"$(nproc)" all
make -j45 all



#prereqs for cuda
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
