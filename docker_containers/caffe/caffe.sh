#!/bin/bash

cd /root
git clone https://github.com/BVLC/caffe.git
cd caffe

# Install python dependencies
cat python/requirements.txt | xargs -n1 pip install && \

# Make and move into build directory
mkdir build && cd build && \

# CMake
cmake .. && \

# Make
#make -j"$(nproc)" all
make -j45 all
