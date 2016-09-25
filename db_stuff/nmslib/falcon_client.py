from jaweson import msgpack
import requests

SEARCH_SERVER = "http://extremeli-evolution-dev-2:8080/test" # use the name of the server running the gunicorn


def test_top_nmslib(fp, k):
    data = msgpack.dumps({"fp": fp,
                          "k": k})
    resp = requests.post(SEARCH_SERVER, data=data)
    return msgpack.loads(resp.content)
