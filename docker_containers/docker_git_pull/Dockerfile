# for docker pycharm integration -  https://plugins.jetbrains.com/plugin/7724?pr=
FROM test:latest

RUN  echo 'helloworld' > test.txt

#good practice to combine update and install to bust cache on the update if the install line changes
#Using -y is convenient to automatically answer yes to all the questions.
#ssh certificate monkey business required to avod interactive password entry on git clone
# i am currently trying ssh-keygen without passphrase
#http://slash-dev-blog.me/docker-git.html
#http://stackoverflow.com/questions/29929534/docker-error-unable-to-locate-package-git

RUN apt-get update && apt-get install -y ca-certificates git-core ssh

#see https://confluence.atlassian.com/pages/viewpage.action?pageId=270827678
#to generate default ssh keys

ENV HOME /root
# add custom ssh config / keys to the root user
ADD ssh/ /root/.ssh/
# Fixes permission if needed
RUN chmod 600 /root/.ssh/*

# Avoid first connection host confirmation
#changed port since i don't think bitbucket supports it
#RUN ssh-keyscan -p2200 bitbucket.com > /root/.ssh/known_hosts
#To improve security, the ssh-keyscan should be run outside the container to fix the known_hosts file once for all.
RUN ssh-keyscan  bitbucket.com > /root/.ssh/known_hosts

RUN mkdir /home/tg
RUN mkdir /home/tg/Dev
WORKDIR /home/tg/Dev

RUN git clone git@bitbucket.org:trendiGuru/core.git
