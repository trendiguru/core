__author__ = 'liorsabag'

from jaweson import msgpack
import requests

#
CLASSIFIER_ADDRESS = "http://37.58.101.170:8080/pd"  # Braini5
CLASSIFIER_ADDRESS = "http://37.58.101.173:8082/pd"  # Braini2


def pd(image_arrary_or_url):
    data = msgpack.dumps({"image": image_arrary_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data)
    return msgpack.loads(resp.content)
