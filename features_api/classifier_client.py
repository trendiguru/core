from jaweson import msgpack
import requests
from .feature_config import FEATURES

def get(feature, image_or_url, **kwargs):
    data = msgpack.dumps({"image_or_url": image_or_url}.update(kwargs))
    resp = requests.post(FEATURE[feature]["url"], data=data)
    return msgpack.loads(resp.content)
