1. Complete a feature and update config.py
2. go to braini2 and open a new screen: screen -S feature_name_falcon
3. Find a server (choose a four digits number that isn't already taken - can check the the ports being used in deployment_config.py) and run `gunicorn -b :<SOME_FREE_PORT> trendi.features_api.app:api --env FEATURES_JSON='["collar"]' -w 2`
  - You can theoretically run multiple classifiers through one gunicorn server in parallel (not tested, probably memory issues)
4. Add entry in deployment_config.py (for documentation)
5. Use, by calling `from trendi.features_api import classifier_client; result = classifier_client.get("collar", img_or_url)`
