from jaweson import msgpack
import requests

#
CLASSIFIER_ADDRESS = "http://37.58.101.173:8080/nd"


def pd(image_array_or_url):
    data = msgpack.dumps({"image": image_array_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data)
    return msgpack.loads(resp.content)
