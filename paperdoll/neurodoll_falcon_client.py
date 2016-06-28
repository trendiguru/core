from jaweson import msgpack
import requests

#
CLASSIFIER_ADDRESS = "http://37.58.101.173:8080/nd"


def pd(image_array_or_url, category_index):
    data = msgpack.dumps({"image": image_array_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data, params={"categoryIndex": category_index})
    return msgpack.loads(resp.content)
