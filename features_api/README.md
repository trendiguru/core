1. Complete a feature and update core.features.config
2. Go to GPU server with our latest docker image (might need to rebuild with to `git -C /usr/lib/python2.7/dist-packages/trendi pull`)
3. Make sure you've copied the PRETRAINED files to the correct locations.
4. Choose a port (choose a four digits number that isn't already taken - can check the the ports being used in deployment_config.py) and run `PORT=<FREE_PORT>; NAME=<RELEVANT_NAME>; nvidia-docker run -d  -v /data:/data -p $PORT:$PORT --name $NAME feature_api:1 bash -c "gunicorn -b :$PORT --env GPU_DEVICE=1 --env FEATURES_JSON='[\"$NAME\"]' -k gevent -w 3 -n $NAME --timeout 120 trendi.features_api.app:api"`

5. Add entry in deployment_config.py (for documentation)
6. Use, by calling `from trendi.features_api import classifier_client; result = classifier_client.get("collar", img_or_url)`


n.b. currently gender is started differently , namely by 

gunicorn -b :8357 -w 5 -k gevent -n gender trendi.features_api.gender_app:api

and also it needs dlib, so you need to do pip install dlib in the container


to add a worker you can do 

docker exec sleeve_length kill -TTIN $(docker exec sleeve_length pgrep -f master)

to kill a worker you can do 

docker exec sleeve_length kill -TTOU $(docker exec sleeve_length pgrep -f master)
