from jaweson import msgpack
import requests

from trendi import constants

CLASSIFIER_ADDRESS = constants.FRCNN_CLASSIFIER_ADDRESS # "http://13.82.136.127:8082/hls"

def detect(img_arr, roi=[]):
    print('using addr '+str(CLASSIFIER_ADDRESS))
    data = {"image": img_arr}
    if roi:
        print "Make sure roi is a list in this order [x1, y1, x2, y2]"
        data["roi"] = roi
    serialized_data = msgpack.dumps(data)
    resp = requests.post(CLASSIFIER_ADDRESS, data=serialized_data)
    return msgpack.loads(resp.content)
