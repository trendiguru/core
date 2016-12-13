1. Complete a feature and update core.features.config
2. Go to any GPU server and open a new screen: screen -S feature_name_falcon
3. Make sure to `git -C /usr/lib/python2.7/dist-packages/trendi pull` and that you've copied the PRETRAINED files to the correct locations.
4. Choose a port (choose a four digits number that isn't already taken - can check the the ports being used in deployment_config.py) and run `gunicorn -b :<SOME_FREE_PORT> --env FEATURES_JSON='["collar"]' --env GPU_DEVICE=<0,1,2,3> -k gevent -w 3 --timeout 120 trendi.features_api.app:api`
5. Add entry in deployment_config.py (for documentation)
6. Use, by calling `from trendi.features_api import classifier_client; result = classifier_client.get("collar", img_or_url)`
