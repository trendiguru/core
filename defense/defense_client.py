from jaweson import msgpack
import requests

from trendi import constants

CLASSIFIER_ADDRESS = constants.FRCNN_CLASSIFIER_ADDRESS # "http://13.82.136.127:8082/hls"

def detect(image_or_url):
    data = msgpack.dumps({"image": image_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data)
    return msgpack.loads(resp.content)
