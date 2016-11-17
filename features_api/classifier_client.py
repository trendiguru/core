from jaweson import msgpack
import requests
from .feature_config import FEATURES         

def feature_url(feature):
    return FEATURES[feature]["server"].rstrip('/') + "/" + feature

for f in FEATURES:
    f["url"] = feature_url(f)

def get(feature, image_or_url, **kwargs):
    data = msgpack.dumps({"image_or_url": image_or_url}.update(kwargs))
    resp = requests.post(FEATURE[feature]["url"], data=data)
    return msgpack.loads(resp.content)


