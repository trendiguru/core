__author__ = 'liorsabag'

from jaweson import msgpack
import requests

#
CLASSIFIER_ADDRESS = "http://37.58.101.170:8080/pd"


def pd(image_arrary_or_url, face):
    data = msgpack.dumps({"image": image_arrary_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data)
    return msgpack.loads(resp.content)
