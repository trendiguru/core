
FROM ubuntu:14.04


#REDIS REDIS REDIS
#install redis
RUN         apt-get update && apt-get install -y redis-server
EXPOSE 6379
ENTRYPOINT  ["/usr/bin/redis-server"]
#alternatively, if we dont want server in container
#RUN pip install redis

