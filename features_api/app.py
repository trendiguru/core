import falcon
import os
import os.path
import json
import pkgutil
from importlib import import_module
from .classifier_falcon import Classifier

# Rely on gunicorn --env (--env FOO=1)
# --env FEATURES_PACKAGE=trendi.features
# --env FEATURES_JSON='["collar", "sleeve"]'

features = json.loads(os.environ.get("FEATURES_JSON", "[]"))
feature_package_string = os.environ.get("FEATURE_PACKAGE", "trendi.features")
if features is "*":
    feature_package = import_module(feature_package_string)
    features = pkgutil.iter_modules(os.path.dirname(feature_package.__file__))

api = falcon.API()
for f in features:
    fpkg = import_module("." + f, feature_package_string)
    if hasattr(fpkg, 'execute') and hasattr(fpkg, 'distance'):
        api.add_route('/{0}'.format(f), Classifier(f, feature_package_string))
