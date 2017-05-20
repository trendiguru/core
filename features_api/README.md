1. Complete a feature and update core.features.config
2. Go to GPU server with our latest docker image (might need to rebuild with to `git -C /usr/lib/python2.7/dist-packages/trendi pull`)
3. Make sure you've copied the PRETRAINED files to the correct locations.
4. Choose a port (choose a four digits number that isn't already taken - can check the the ports being used in deployment_config.py) and run `PORT=<FREE_PORT>; NAME=<RELEVANT_NAME>; nvidia-docker run -d  -v /data:/data -p $PORT:$PORT --name $NAME feature_api:2 bash -c "gunicorn -b :$PORT --env GPU_DEVICE=1 --env FEATURES_JSON='[\"$NAME\"]' -k gevent -w 3 -n $NAME --timeout 120 trendi.features_api.app:api"`

5. Add entry in deployment_config.py (for documentation)
6. Use, by calling `from trendi.features_api import classifier_client; result = classifier_client.get("collar", img_or_url)`


n.b. currently gender is started differently , namely by 

gunicorn -b :8357 -w 5 -k gevent -n gender trendi.features_api.gender_app:api

and also it needs dlib, so you need to do pip install dlib in the container


To add a worker:

`NAME=sleeve_length; docker exec $NAME kill -TTIN $(docker exec $NAME pgrep -f master)`

To kill a worker you can do:

`NAME=sleeve_length; docker exec $NAME kill -TTOU $(docker exec $NAME pgrep -f master)`



DRIVER ISSUES
if you need to roll back a driver e.g. if nvidia-smi gives "Failed to initialize NVML: Driver/library version mismatch"
first maybe just rebuild the docker container which apparently expects a particular driver

if that doesnt work:
1. if cuda is not installed, install it, e.g.
sudo sh cuda_8.0.44_linux-run
do this first since it installs the latest drivers it seems

2. kill newest driver e..g
sudo apt-get remove nvidia-375

3. install new driver e.g.
sudo sh NVIDIA-Linux-x86_64-367.57.run

4. check nvidia-smi runs and shows right version

5. run new containers as in step 4 above e.g.
`PORT=<FREE_PORT>; NAME=<RELEVANT_NAME>; nvidia-docker run -d  -v /data:/data -p $PORT:$PORT --name $NAME feature_api:1 bash -c "gunicorn -b :$PORT --env GPU_DEVICE=1 --env FEATURES_JSON='[\"$NAME\"]' -k gevent -w 3 -n $NAME --timeout 120 trendi.features_api.app:api"`
which actually didnt run in one go for me so i had t break it into two parts, interactive run and then gunicorn
PORT=<FREE_PORT>; NAME=<RELEVANT_NAME>; nvidia-docker run -it  -v /data:/data -p $PORT:$PORT --name $NAME feature_api:1 bash
and then
gunicorn -b :$PORT --env GPU_DEVICE=1 --env FEATURES_JSON=[\"$NAME\"] -k gevent -w 3 -n $NAME --timeout 120 trendi.features_api.app:api"
