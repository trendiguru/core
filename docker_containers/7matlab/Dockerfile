
#to run gui app (giving up since i can't seem to install license manager - product not given as an option in install....)
FROM ubuntu:14.04

RUN apt-get update && apt-get install -y firefox

# Replace 1000 with your user / group id
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer


#matlab install needs java
RUN apt-get install -y default-jre


USER developer
ENV HOME /home/developer
CMD /usr/bin/firefox


#somethign to try - just link -v to matlab files and run install

#to run this:
#docker build -t firefox . it and run the container with:

#
#docker run -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority --net=host firefox