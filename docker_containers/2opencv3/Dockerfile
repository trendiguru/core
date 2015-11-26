#try copying /etc/apt/sources.list into the docker to get the installs to work
FROM pip
#FROM ubuntu:14.04
#alternatively, ubuntu:latest

#vars

#attempt at full on trendiguru in container -
#opencv, web2py, redis, and our git repo
#since our mongodb is large i am not including that
#this is supposed to be a 'tg instance' which has all necessary code to do tg stuff

#question 1
#do we want web2py in the container or do we rely on web2py running on extremeli
#i will try in container. This requires apache/whatever other server(s) we run too

#question 2
#do we want mongodb in the container or do we rely on extremeli ?
#due to size, keep db on extremeli but install in container so that mongo function calls work
#so all we need in the container is the pymongo install , using pip

#question 3
#redis? i will try it in the container
#alternate would be to run redis on extremli and then we'd only need pip install redis or the like

#our repo, can be accessed using 'docker run -v' option
#to link local repo directory (which can be modified by pycharm or whatever the coder wants)
#to a directory in the container
#in any case i've not seen any way to go the other direction (allow external access to container files)

MAINTAINER Jeremy Rutman <jeremy@trendiguru.com>



#OPENCV OPENCV OPENCV
#install opencv3.0
#RUN apt-get install -y libgtk2.0-dev
#RUN apt-get -qq install -y libopencv-dev
RUN apt-get -qq install -y build-essential
RUN apt-get -qq install -y checkinstall
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
RUN apt-get -qq install -y libqt4-dev
RUN apt-get -qq install -y libgtk2.0-dev
RUN apt-get -qq install -y libfaac-dev
RUN apt-get -qq install -y libmp3lame-dev
RUN apt-get -qq install -y libopencore-amrnb-dev
RUN apt-get -qq install -y libopencore-amrwb-dev
RUN apt-get -qq install -y libtheora-dev
RUN apt-get -qq install -y libvorbis-dev
RUN apt-get -qq install -y libxvidcore-dev
RUN apt-get -qq install -y x264
RUN apt-get -qq install -y v4l-utils
RUN apt-get -qq install -y ffmpeg

# libtiff4-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine-dev libgstreamer0.10-dev
#libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip


#install opengl
RUN apt-get install -y freeglut3-dev
RUN  mkdir -p /opt/OpenCV
RUN  cd /opt/OpenCV
RUN pwd
WORKDIR /opt/OpenCV
RUN pwd

# test on jr's pc
wget http://sourceforge.net/projects/opencvlibrary/files/latest/download/opencv-3.0.0.zip

RUN  wget -O OpenCV3.0.zip https://codeload.github.com/Itseez/opencv/zip/3.0.0
RUN  unzip OpenCV3.0.zip
RUN  wget -O opencv_contrib.zip https://codeload.github.com/Itseez/opencv_contrib/zip/3.0.0
RUN  unzip opencv_contrib.zip

ENV OPENCV_HOME /opt/OpenCV

#add the nonfree stuff here if necessary
RUN  mkdir -p $OPENCV_HOME/opencv-3.0.0/build
RUN  cd $OPENCV_HOME/opencv-3.0.0/build
RUN pwd
WORKDIR /opt/OpenCV/opencv-3.0.0/build
RUN pwd

#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..
#make -j $(nproc)
#sudo make install
#sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
#sudo ldconfig


#for fisherfaces: -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_directory>
#apparently this actually requires dl of a further repo at https://github.com/Itseez/opencv_contrib.git
RUN  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") -D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF -D BUILD_opencv_java=OFF -D WITH_IPP=OFF -D OPENCV_EXTRA_MODULES_PATH=/OpenCV/opencv_contrib-3.0.0/modules -D BUILD_NEW_PYTHON_SUPPORT=ON  -DWITH_QT=ON -DWITH_OPENGL=ON -DWITH_TBB=ON -D WITH_V4L=ON ..
#RUN  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D BUILD_opencv_nonfree=ON ..
RUN  make -j2
RUN  make install
#RUN  make clean

RUN  sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN  ldconfig
RUN  echo "OpenCV 3.0.0 is ready to be used"
RUN export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH


# get GUI connection
# Replace 1000 with your user / group id
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer

USER developer
ENV HOME /home/developer

#to run this:
#docker run -e DISP containername