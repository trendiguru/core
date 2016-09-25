from jaweson import msgpack
import requests

SERVER = "http://extremeli-evolution-dev-2:8080/"  # use the name of the server running the gunicorn


def nmslib_find_top_k(fp, k , category):
    data = msgpack.dumps({"fp": fp,
                          "k": k})
    category_server = SERVER+category
    resp = requests.post(category_server, data=data)
    return msgpack.loads(resp.content)
