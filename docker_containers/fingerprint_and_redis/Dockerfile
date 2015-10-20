FROM ubuntu:14.04
ENV n_procs $(cat /proc/cpuinfo | grep processor | wc -l)
RUN echo n_procs
RUN echo ${cat /proc/cpuinfo | grep processor | wc -l}
RUN mkdir ${cat /proc/cpuinfo | grep processor | wc -l}

RUN apt-get install -y wget
RUN apt-get install -y python
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install rq

CMD rqworker -u redis://redis1-redis-1-vm:6379 testqueue &