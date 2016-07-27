from jaweson import msgpack
import requests

#
CLASSIFIER_ADDRESS = "http://37.58.101.173:8080/nd"


def pd(fp, col_name, category):
    params = {"category": category,
              "col_name" : col_name}
    data = msgpack.dumps({"fingerprint": fp})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data, params=params)
    return msgpack.loads(resp.content)