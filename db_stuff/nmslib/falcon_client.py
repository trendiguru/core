from jaweson import msgpack
import requests

SERVER = "http://extremeli-evolution-dev-2:"  # use the name of the server running the gunicorn


def nmslib_find_top_k(fp, k, port,category):
    data = msgpack.dumps({"fp": fp,
                          "k": k})
    category_server = SERVER+port+'/'+category
    resp = requests.post(category_server, data=data)
    return msgpack.loads(resp.content)
