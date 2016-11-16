import falcon
import os.environ
import os.path
import json
import pkgutil
import testpkg
from importlib import import_module

# Rely on gunicorn --env (--env FOO=1)
# --env FEATURE_PACKAGE=trendi.features

feature_package_string = os.environ.get("FEATURE_PACKAGE", "trendi.features")
feature_package = import_module(feature_package_string)

iter_features = pkgutil.iter_modules(os.path.dirname(feature_package.__file__))
api = falcon.API()
for f in iter_features:
    fpkg = import_module("." + f, feature_package_string)
    if hasattr(fpkg, 'execute') and hasattr(fpkg, 'distance'):
        api.add_route('/{0}'.format(f), Classifier(f, feature_package_string))
