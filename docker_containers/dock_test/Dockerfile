FROM  ubuntu:14.04

MAINTAINER Jeremy Rutman <jeremy@trendiguru.com>


#SSH SSH SSH
RUN apt-get update
RUN apt-get install -y python
RUN apt-get install -y build-essential
RUN apt-get install -y make
RUN apt-get install -y gcc
RUN apt-get install -y ssh


RUN echo hi
CMD ssh -i /.ssh/first_aws.pem -L 27017:localhost:27017 ubuntu@extremeli.trendi.guru && /bin/bash


# see http://redis.io/download
RUN wget http://download.redis.io/releases/redis-3.0.4.tar.gz
RUN tar xzf redis-3.0.4.tar.gz
WORKDIR redis-3.0.4
RUN ls -ltr
RUN make


#these scripts are to check port forwarding with a single command
RUN  echo 'nmap -p 27017 localhost' > portmongo.sh
RUN chmod o+x portmongo.sh
RUN  echo 'nmap -p 6379 localhost' > portredis.sh
RUN chmod o+x portredis.sh

#potentially useful


