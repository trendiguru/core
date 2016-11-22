from jaweson import msgpack
import requests
from .deployment_config import DEPLOYMENTS

def get(feature, image_or_url, **kwargs):
    kwargs.update({"image_or_url": image_or_url})
    data = msgpack.dumps(kwargs)
    resp = requests.post(DEPLOYMENTS[feature]["url"], data=data)
    return msgpack.loads(resp.content)
