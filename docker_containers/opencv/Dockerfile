FROM ubuntu:14.04


##############################
#OPENCV
##############################


### OPENCV 3.0 ###
### OPENCV 3.0 ###
### OPENCV 3.0 ###

MAINTAINER Jeremy Rutman <jeremy@trendiguru.com>

#OPENCV OPENCV OPENCV
#install opencv3.0
#RUN apt-get install -y libgtk2.0-dev
#RUN apt-get -qq install -y libopencv-dev
RUN apt-get -qq install -y build-essential
#RUN apt-get -qq install -y checkinstall
RUN apt-get -qq install -y cmake
RUN apt-get -qq install -y pkg-config
RUN apt-get -qq install -y yasm
#RUN apt-get -qq install -y libjpeg-dev
RUN apt-get -qq install -y libpng12-dev
#RUN apt-get -qq install -y libjasper-dev
#RUN apt-get -qq install -y libavcodec-dev
#RUN apt-get -qq install -y libavformat-dev
#RUN apt-get -qq install -y libswscale-dev
#RUN apt-get -qq install -y libdc1394-22
#RUN apt-get -qq install -y libdc1394-22-dev
#RUN apt-get -qq install -y libxine-dev
#RUN apt-get -qq install -y libgstreamer0.10-dev
#RUN apt-get -qq install -y libgstreamer-plugins-base0.10-dev
#RUN apt-get -qq install -y libv4l-dev
RUN apt-get -qq install -y python-dev
RUN apt-get -qq install -y python-numpy
#RUN apt-get -qq install -y libtbb-dev
# runs into error 100  The command '/bin/sh -c apt-get -qq install -y libqt4-dev' returned a non-zero code: 100
#RUN apt-get -qq install -y libqt4-dev
#RUN apt-get -qq install -y libgtk2.0-dev
#RUN apt-get -qq install -y libfaac-dev
#RUN apt-get -qq install -y libmp3lame-dev
#RUN apt-get -qq install -y libopencore-amrnb-dev
#RUN apt-get -qq install -y libopencore-amrwb-dev
#RUN apt-get -qq install -y libtheora-dev
#RUN apt-get -qq install -y libvorbis-dev
#RUN apt-get -qq install -y libxvidcore-dev
#RUN apt-get -qq install -y x264
#RUN apt-get -qq install -y v4l-utils
#RUN apt-get -qq install -y ffmpeg

# libtiff4-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine-dev libgstreamer0.10-dev
#libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip

#QT5
#RUN apt-get install -y perl wget
#RUN apt-get install -y libGL-dev

#RUN mkdir /inst && cd /inst && wget \
#  http://download.qt.io/official_releases/qt/5.4/5.4.0/single/qt-everywhere-opensource-src-5.4.0.tar.gzf

#RUN mkdir /src && cd /src && tar -xvf /inst/qt-everywhere-opensource-src-5.4.0.tar.gz
#RUN mv /src/qt-everywhere-opensource-src-5.4.0 /src/qt
#RUN cd /src/qt && ./configure \
#  -confirm-license -opensource \
#  -nomake examples -nomake tests -no-compile-examples \
#  -no-xcb \
#  -prefix "/usr/local/Qt"
#RUN cd /src/qt && make -j4 all


#install opengl
#RUN apt-get install -y freeglut3-dev
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

#RUN  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON \
# -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D BUILD_opencv_nonfree=ON ..

RUN  make -j8 | tee makeout.txt
RUN  make install | tee installout.txt

RUN  sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN  ldconfig
RUN  echo "OpenCV 3.0.0 is ready to be used"
RUN export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH
