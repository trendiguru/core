#attempt at full on trendiguru in container -
#opencv, mongodb, web2py, redis, and our git repo
#our repo, i thnk, shouldbe accessed using 'docker run -v' option
#to link local repo directory (which can be modified by pycharm or whatever the coder wants)
#to a directory in the container
#in any case i've not seen any way to go the other direction (allow external access to container files)

FROM ubuntu:14.04
#alternatively, ubuntu:latest

MAINTAINER Jeremy Rutman <jeremy@trendiguru.com>

#PYTHON NUMPY
#install python2.7, numpy
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential checkinstall cmake pkg-config yasm unzip wget && \
  apt-get install -y python-dev python-numpy python-dev python-numpy && \
  apt-get autoclean && \
  rm -rf /var/lib/apt/lists/*

#OPENCV OPENCV OPENCV
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


#REDIS REDIS REDIS
#install redis
RUN         apt-get update && apt-get install -y redis-server
EXPOSE 6379
ENTRYPOINT  ["/usr/bin/redis-server"]


#MONGODB MONGODB MONGODB
#install mongodb
# Installation:
# Import MongoDB public GPG key AND create a MongoDB list file
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10
RUN echo "deb http://repo.mongodb.org/apt/ubuntu "$(lsb_release -sc)"/mongodb-org/3.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-3.0.list
RUN apt-get update && apt-get install -y mongodb-org
#or if we need particular version:
#RUN apt-get update && apt-get install -y mongodb-org=3.0.1 mongodb-org-server=3.0.1 mongodb-org-shell=3.0.1 mongodb-org-mongos=3.0.1 mongodb-org-tools=3.0.1
#make mongodb data directory: (mkdir -p adds parents as necessary, and no error if dir exists)
RUN mkdir -p /data/db
# Expose port 27017 from the container to the host
EXPOSE 27017
#entrypoint makes the container run as an executable
#ENTRYPOINT ["/usr/bin/mongod"]
#docker run -p 27017:27017 --name mongo_instance_001 -d my/repo

#WEB2PY WEB2PY WEB2PY
#install web2py , from http://www.web2pyslices.com/slice/show/2014/web2py-dockerfile
## Add application resources URL to sources list
#RUN echo "deb http://archive.ubuntu.com/ubuntu/ $(lsb_release -sc) main universe" >> /etc/apt/sources.list
## Update the sources list
#RUN apt-get update

## Install basic Python tools
RUN apt-get install -y python python-dev python-setuptools python-pip

## Install git
RUN apt-get install -y git

## Install gunicorn
#RUN apt-get install -y gunicorn

## Install virtualenv
#RUN pip install virtualenv

## Build directory for web2py app
RUN mkdir /var/www

## Build virtual environment
#RUN virtualenv /var/www/env

## Clone web2py directory
RUN git clone --recursive http://github.com/web2py/web2py.git /var/www/web2py

## Expose ports
EXPOSE 80

## Set default working directory where CMD will execute
WORKDIR /var/www/web2py

## Set the default command to execute when creating a new container
#CMD . /var/www/env/bin/activate
#CMD python anyserver.py -s gunicorn -i 0.0.0.0 -p 80
