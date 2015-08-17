
#ok one way to get this running seems to be to run from shell , then do the ssh port forward, then
#docker exec -it <name> bash
#if you do that and check port 27017 you will find port running


#run this (assuming image is built to name mongotest) by:
#docker run -v /home/jeremy/.ssh:/ssh -it mongotest /bin/bash

FROM tg2
RUN apt-get update
#not sure if build-essential is essential
RUN apt-get install -y build-essential


#PYTHON STUFF
RUN apt-get install -y build-essential python-dev
RUN apt-get install -y wget
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install  pymongo
RUN pip install  numpy


RUN apt-get install -y ca-certificates ssh
#RUN apt-get install -y screen
#RUN screen -dRR mongo
#this config prevents the 'are you sure prompt' from spoiling the cabbage
ADD ssh_config /etc/ssh/ssh_config
RUN mkdir -p /.ssh

ADD ssh/first_aws.pem /.ssh/first_aws.pem

EXPOSE 27017
CMD ssh -i /.ssh/first_aws.pem -L 27017:localhost:27017 ubuntu@extremeli.trendi.guru && /bin/bash

 #tail -F /var/log/booterror.log


#CMD service mysql start && tail -F /var/log/mysql/error.log
#or use foreground command to do this
#CMD /usr/bin/mysqld_safe
#or wrap your scripts into start.sh and put this in end
#CMD /start.sh

#RUN ls ssh




#teum mas esti 8292317/2