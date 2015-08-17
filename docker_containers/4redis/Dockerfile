
FROM tg3


#REDIS REDIS REDIS
#install redis
RUN         apt-get update && apt-get install -y redis-server
EXPOSE 6379

#don't do this in brackets cuz it wont work , this is sposed to be fixed in next docker version
ENTRYPOINT  "/usr/bin/redis-server"


#alternatively, if we dont want server in container
#RUN pip install redis

