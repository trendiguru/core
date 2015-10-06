FROM ubuntu:14.04

###############
#BASICS
###############
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
RUN apt-get install -y screen
#nmap is  for debugging port forwarding
RUN apt-get install -y nmap
#RUN apt-get install -y checkinstall
RUN apt-get install -y cmake
RUN apt-get install -y unzip
RUN apt-get install -y python-tk
RUN apt-get -y --force-yes  install nano

##########
#PYTHON ANACONDA - numpy,matplotlib, plus more sci stuff - if this works remove matplotlib, numpy etc elsewhere in dockerfile
##########
RUN wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.3.0-Linux-x86_64.sh
RUN bash Anaconda-2.3.0-Linux-x86_64.sh

##############
#PYTHON NUMPY  - dont need this if anaconda is done
##############
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


#####################
#get GUI connection
#####################
# Replace 1000 with your user / group id
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer


#

#####################
# ssh port forwarding
#####################
#turn off asking 'are you sure' on first ssh connect
#this is supposed to be taken care of by the keyscan but isn't , perhaps because this
# is not our own repo running on a port like in the example
ADD ssh/ssh_config /etc/ssh/ssh_config

# Fixes permission if needed
#RUN chmod 600 /.ssh/*
RUN chmod 600 /home/developer/.ssh/*
RUN chown developer:developer /home/developer/
ADD ssh/first_aws.pem /home/developer/.ssh/first_aws.pem
RUN chown developer:developer /home/developer/.ssh/*
RUN chmod 400 /home/developer/.ssh/*

# Avoid first connection host confirmation
RUN ssh-keyscan  bitbucket.com > /home/developer/.ssh/known_hosts

#mongobd port forward
#CMD ssh -i /.ssh/first_aws.pem -L 27017:localhost:27017 ubuntu@extremeli.trendi.guru && /bin/bash
RUN ssh -N -L 6379:localhost:6379 ubuntu@extremeli.trendi.guru &

#these scripts are to check port forwarding with a single command
RUN  echo 'nmap -p 27017 localhost' > portmongo.sh
RUN chmod o+x portmongo.sh
RUN  echo 'nmap -p 6379 localhost' > portredis.sh
RUN chmod o+x portredis.sh

################################
# git clone the tg repo
################################
RUN apt-get update
#see https://confluence.atlassian.com/pages/viewpage.action?pageId=270827678
#to generate default ssh keys
RUN apt-get install -y ca-certificates
#RUN apt-get install -y git-core
RUN apt-get install -y --force-yes git
RUN apt-get install -y ssh
RUN adduser develop   #check that this works
ENV HOME /home/developer
# add custom ssh config / keys to the root user
#ADD ssh/ /root/.ssh/
ADD ssh/ /home/developer/.ssh/
RUN mkdir /home/developer/tg
RUN chown developer:developer /home/developer/tg
RUN mkdir /home/developer/tg/Dev
RUN chown developer:developer /home/developer/tg/Dev
WORKDIR /home/developer/tg/Dev
USER developer
RUN whoami
RUN git clone --verbose git@bitbucket.org:trendiGuru/core.git
#maybe ssh is per user so do this as developer

WORKDIR /home/developer/tg/Dev/core
RUN ssh -N -L 6379:localhost:6379 ubuntu@extremeli.trendi.guru &
RUN rqworker gender &


#advice from docker forum guy
#RUN yum install -y openssh-clients
#COPY id_rsa /tmp/id_rsa
#CMD ssh  -i /tmp/id_rsa -L 8000:localhost:8000 -N vagrant@192.168.33.10 -o StrictHostKeyChecking=no

CMD ssh  -L 6379:localhost:6379 -N  -o StrictHostKeyChecking=no -f ubuntu@extremeli.trendi.guru &
