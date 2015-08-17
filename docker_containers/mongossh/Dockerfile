#run this (assuming image is built to name mongotest) by:
#docker run -v /home/jeremy/.ssh:/ssh -it mongotest /bin/bash

FROM ubuntu:14.04

RUN apt-get install -y ca-certificates ssh
#RUN apt-get install -y screen
#RUN screen -dRR mongo
#this config prevents the 'are you sure prompt' from spoiling the cabbage
ADD ssh_config /etc/ssh/ssh_config

CMD ssh -i /ssh/first_aws.pem -L 27017:localhost:27017 ubuntu@extremeli.trendi.guru
#RUN ls ssh
