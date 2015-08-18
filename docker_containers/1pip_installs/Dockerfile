FROM ubuntu:14.04

RUN apt-get update
RUN apt-get -y upgrade
#not sure if build-essential is essential
RUN apt-get install -y build-essential
RUN apt-get install -y pkg-config
RUN apt-get install -y yasm
RUN apt-get install -y wget
RUN apt-get install -y  libpng-dev
RUN apt-get install -y  zlib1g-dev
RUN apt-get install -y freetype*
RUN apt-get install -y python-tk
RUN apt-get install -y screen
#this is  for debugging port forwarding
RUN apt-get install -y nmap
RUN apt-get install -y checkinstall
RUN apt-get install -y cmake
RUN apt-get install -y unzip

#PYTHON NUMPY
RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list
#not sure why the below is used
#RUN checkinstall cmake pkg-config yasm unzip wget
RUN apt-get install -y python-dev
RUN apt-get install -y python-numpy
RUN apt-get autoclean
RUN rm -rf /var/lib/apt/lists/*
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install pymongo
RUN pip install ipython

RUN pip install numpy
RUN pip install python-dateutil
RUN pip install pyparsing
RUN pip install pytz
RUN pip install matplotlib

#so you can check port forwarding with a single command
RUN  echo 'nmap -p 27017 localhost' > portmongo.sh
RUN chmod o+x portmongo.sh
RUN  echo 'nmap -p 6379 localhost' > portredis.sh
RUN chmod o+x portredis.sh

#potentially useful
#sudo apt-get install python-memcache
#sudo pip install genderizer
#sudo pip install  naiveBayesClassifier
