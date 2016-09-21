from jaweson import msgpack
import requests

SEARCH_SERVER = "http://104.155.77.106:8080/test"


def test_top_nmslib(fp):
    data = msgpack.dumps({"fp": fp})
    resp = requests.post(SEARCH_SERVER, data=data)
    return msgpack.loads(resp.content)
