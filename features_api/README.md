1. Complete a feature and place it in core/features
2. Find a server and run `gunicorn -b :<SOME_FREE_PORT> trendi.features_api.app:api --env FEATURES_JSON='["collar"]' -w 2`
  - You can theoretically run multiple classifiers through one gunicorn server in parallel (not tested, probably memory issues)
3. Add entry in classifier_client.FEATURES
4. Use, by calling `from trendi.features_api import classifier_client; result = classifier_client.get("sleeve", img_url)`
