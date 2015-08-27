# sshd, matlab, dcmdump (via dcmtk)

FROM ubuntu:12.04

ADD matlab.txt /mcr-install/matlab.txt
ADD id_rsa.pub /root/.ssh/authorized_keys
ADD assets /root/assets


RUN \
   apt-get update && \
   apt-get install -y curl wget xorg unzip openssh-server dcmtk && \
   mkdir /var/run/sshd

RUN \
   cd /mcr-install && \

#this dockerfile originally got the installer with a wget
#   wget -nv http://www.mathworks.de/supportfiles/downloads/R2013b/deployment_files/R2013b/installers/glnxa64/MCR_R2013b_glnxa64_installer.zip && \
#but since i can't find the actual remote file I add it locally - requiring the docker user to download it :<<

ADD matlab_R2015a_glnxa64.zip /home/jeremy/Downloads/matlab_R2015a_glnxa64.zip

RUN unzip matlab_R2015a_glnxa64.zip
RUN mkdir /opt/mcr && \
RUN   ./install -inputFile matlab.txt
RUN   cd /
RUN   rm -rf mcr-install

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]