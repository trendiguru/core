FROM ubuntu:14.04
#alternatively, ubuntu:latest

MAINTAINER Jeremy Rutman <jeremy@trendiguru.com>


#SSH SSH SSH
#not sure what ca-certificates is
RUN apt-get install -y ca-certificates ssh
RUN apt-get install -y screen
RUN screen -dRR mongo
RUN 'ssh -i first_aws.pem -L 27017:localhost:27017 ubuntu@extremeli.trendi.guru'

