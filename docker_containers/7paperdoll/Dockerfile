FROM tg3
#FROM ubuntu:14.04

#cheatsheet https://github.com/yaronr/docker-cheat-sheet
#to access ports use -P option to port forward ports thru container wall, like
#docker run -d -P training/webapp python app.py
#docker run -d -p 8000:8000  --link pg:database -t django_app

#to use local filesystem file in a docker container use -v:
#docker run -d --name newguest3 -P -v /var/www/html/index.html:/var/www/html/index.html:ro mymod/httpd:v2
#ro is read only - see http://docs.oracle.com/cd/E52668_01/E54669/html/section_vtr_p2z_fp.html

#linking containers
#docker run -d -p 8000:8000  --link pg:database -t django_app

# for docker pycharm integration -  https://plugins.jetbrains.com/plugin/7724?pr=



RUN  echo 'helloworld' > test.txt

#good practice to combine update and install to bust cache on the update if the install line changes
#Using -y is convenient to automatically answer yes to all the questions.
#ssh certificate monkey business required to avod interactive password entry on git clone
# i am currently trying ssh-keygen without passphrase
#http://slash-dev-blog.me/docker-git.html
#http://stackoverflow.com/questions/29929534/docker-error-unable-to-locate-package-git
#docker run

RUN apt-get update
RUN apt-get install -y ca-certificates
#RUN apt-get install -y git-core
RUN apt-get install -y --force-yes git
RUN apt-get install -y ssh

#see https://confluence.atlassian.com/pages/viewpage.action?pageId=270827678
#to generate default ssh keys

RUN adduser testuser
#USER testuser
#RUN whoami

ENV HOME /home/testuser
# add custom ssh config / keys to the root user
#ADD ssh/ /root/.ssh/
ADD ssh/ /home/testuser/.ssh/

#turn off asking 'are you sure' on first ssh connect
#this is supposed to be taken care of by the keyscan but isn't , perhaps because this
# is not our own repo running on a port like in the example
ADD ssh/ssh_config /etc/ssh/ssh_config

# Fixes permission if needed
RUN chmod 600 /.ssh/*
RUN chmod 600 /home/testuser/.ssh/*
RUN chown testuser:testuser /home/testuser/
ADD ssh/first_aws.pem /home/testuser/.ssh/first_aws.pem
RUN chown testuser:testuser /home/testuser/.ssh/*

# Avoid first connection host confirmation
#changed port since i don't think bitbucket supports it
#RUN ssh-keyscan -p2200 bitbucket.com > /root/.ssh/known_hosts
#To improve security, the ssh-keyscan should be run outside the container to fix the known_hosts file once for all.
#RUN ssh-keyscan  -p2200 bitbucket.com > /home/testuser/.ssh/known_hosts
#RUN ssh-keyscan  bitbucket.com > /root/.ssh/known_hosts
RUN ssh-keyscan  bitbucket.com > /home/testuser/.ssh/known_hosts

RUN mkdir /home/testuser/tg
RUN chown testuser:testuser /home/testuser/tg
RUN mkdir /home/testuser/tg/Dev
RUN chown testuser:testuser /home/testuser/tg/Dev
WORKDIR /home/testuser/tg/Dev

RUN apt-get -y --force-yes  install nano

USER testuser
RUN whoami
RUN git clone --verbose git@bitbucket.org:trendiGuru/core.git
CMD ssh -i /.ssh/first_aws.pem -L 27017:localhost:27017 ubuntu@extremeli.trendi.guru && /bin/bash

