#install opencv

FROM ubuntu:14.04

#install python2.7, numpy
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential checkinstall cmake pkg-config yasm unzip wget && \
  apt-get install -y python-dev python-numpy python-dev python-numpy && \
  apt-get autoclean && \
  rm -rf /var/lib/apt/lists/*

#install opencv3.0
RUN \
  mkdir -p /opt/OpenCV && \
  cd /opt/OpenCV && \
  wget -O OpenCV3.0.zip https://codeload.github.com/Itseez/opencv/zip/3.0.0 && \
  unzip OpenCV3.0.zip && \
  wget -O opencv_contrib.zip https://codeload.github.com/Itseez/opencv_contrib/zip/3.0.0 && \
  unzip opencv_contrib.zip

ENV OPENCV_HOME /opt/OpenCV

#add the nonfree stuff here if necessary
RUN \
  mkdir -p $OPENCV_HOME/opencv-3.0.0/build && \
  cd $OPENCV_HOME/opencv-3.0.0/build && \
  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") -D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF -D BUILD_opencv_java=OFF -D WITH_IPP=OFF -D OPENCV_EXTRA_MODULES_PATH=/OpenCV/opencv_contrib-3.0.0/modules .. && \
#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_opencv_nonfree=ON ..
  make -j8 && \
  make install && \
  make clean
  
RUN \
  sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && \
  ldconfig && \
  echo "OpenCV 3.0.0 is ready to be used"

