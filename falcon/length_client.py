from jaweson import msgpack
import requests

CLASSIFIER_ADDRESS = "http://37.58.101.173:8083/length"


def get_length(image_or_url):
    data = msgpack.dumps({"image_or_url": image_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data)
    return msgpack.loads(resp.content)
