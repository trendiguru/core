#fingerprint_allinone - single dockerfile for:
#pip , ipython, numpy
#opencv3
#git clone our repo

#NOTE I have seen the docker build command break due to momentary internet hiccups.
#you can just run the docker build again and it will try again from where it left off
#to access ports use -P option to port forward ports thru container wall, like
#docker run -d -P training/webapp python app.py
#docker run -d -p 8000:8000  --link pg:database -t django_app

#to use local filesystem file in a docker container use -v:
#docker run -d --name newguest3 -P -v /var/www/html/index.html:/var/www/html/index.html:ro mymod/httpd:v2
#ro is read only - see http://docs.oracle.com/cd/E52668_01/E54669/html/section_vtr_p2z_fp.html

#linking containers
#docker run -d -p 8000:8000  --link pg:database -t django_app

FROM ubuntu:14.04

#increase the available sources for installs
#RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-key update

#not sure if build-essential is essential
RUN apt-get install -y build-essential
RUN apt-get install -y pkg-config
RUN apt-get install -y yasm
RUN apt-get install -y wget
RUN apt-get install -y libpng-dev
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y freetype*
RUN apt-get install -y python-tk
RUN apt-get install -y screen
#cmap is  for debugging port forwarding
RUN apt-get install -y nmap
RUN apt-get install -y checkinstall
RUN apt-get install -y cmake
RUN apt-get install -y unzip

#PYTHON NUMPY
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

#genderizer, bayes is for text analysis - helper func to generate training files
#at this point not necessary
#RUN pip install genderizer
#RUN pip install  naiveBayesClassifier

#these scripts are to check port forwarding with a single command
RUN  echo 'nmap -p 27017 localhost' > portmongo.sh
RUN chmod o+x portmongo.sh
RUN  echo 'nmap -p 6379 localhost' > portredis.sh
RUN chmod o+x portredis.sh

#potentially useful
#sudo apt-get install python-memcache
#sudo pip install genderizer
#sudo pip install  naiveBayesClassifier




