from jaweson import msgpack
import requests

CLASSIFIER_ADDRESS = "http://37.58.101.173:8081/sleeve"


def get_sleeve(image_or_url):
    data = msgpack.dumps({"function": "execute", "image_or_url": image_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data)
    return msgpack.loads(resp.content)


def sleeve_distance(main_vector, candidate_vector):
    data = msgpack.dumps({"function": "distance", 'main_vector': main_vector, 'candidate_vector': candidate_vector})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data)
    return msgpack.loads(resp.content)
