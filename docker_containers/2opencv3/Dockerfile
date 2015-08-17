FROM tg1
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
RUN  mkdir -p /opt/OpenCV
RUN  cd /opt/OpenCV
RUN pwd
WORKDIR /opt/OpenCV
RUN pwd
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

RUN  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") -D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF -D BUILD_opencv_java=OFF -D WITH_IPP=OFF -D OPENCV_EXTRA_MODULES_PATH=/OpenCV/opencv_contrib-3.0.0/modules ..
#RUN  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D BUILD_opencv_nonfree=ON ..
RUN  make -j8
RUN  make install
RUN  make clean

RUN  sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN  ldconfig
RUN  echo "OpenCV 3.0.0 is ready to be used"
RUN export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH