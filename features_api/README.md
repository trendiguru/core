1. Complete a feature and update core.features.config
2. Go to any GPU server and open a new screen: screen -S feature_name_falcon
3. Make sure to `git -C /usr/lib/python2.7/dist-packages/trendi pull` and that you've copied the PRETRAINED files to the correct locations.
4. Choose a port (choose a four digits number that isn't already taken - can check the the ports being used in deployment_config.py) and run `gunicorn -b :<SOME_FREE_PORT> --env FEATURES_JSON='["collar"]' --env GPU_DEVICE=<0,1,2,3> -k gevent -w 3 --timeout 120 trendi.features_api.app:api`
  This all should probably be happening within a docker container so what you can do is something like : 
  
  nvidia-docker run -it -v /data:/data -p 8085:8085 --name collar eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:1 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull &&  gunicorn -b :8085 --env GPU_DEVICE=1 --env FEATURES_JSON='["collar"]' -w 3 -k gevent -n collar --timeout 120 trendi.features_api.app:api'
  
  actually the use of  single quotes within the gunicorn command messes up the single quote that is trying to wrap all the commands to give docker, and its prob. hard to get around this since both single and double quotes are being used. So you can do 
  
  nvidia-docker run -it -v /data:/data -p 8085:8085 --name collar eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:1 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'   
  
  and then finish interactively by running the gunicorn command, namely
  
  gunicorn -b :8085 --env GPU_DEVICE=1 --env FEATURES_JSON='["collar"]' -w 3 -k gevent -n collar --timeout 120 trendi.features_api.app:api


5. Add entry in deployment_config.py (for documentation)
6. Use, by calling `from trendi.features_api import classifier_client; result = classifier_client.get("collar", img_or_url)`


n.b. currently gender is started differently , namely by 

gunicorn -b :8357 -w 5 -k gevent -n gender trendi.features_api.gender_app:api

and also it needs dlib, so you need to do pip install dlib in the container
