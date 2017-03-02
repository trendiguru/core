__author__ = 'liorsabag'

from jaweson import msgpack
import requests
from trendi import constants
import json
#
# CLASSIFIER_ADDRESS = "http://37.58.101.170:8080/pd"  # Braini5
# CLASSIFIER_ADDRESS = "http://37.58.101.173:8082/pd"  # Braini2
# CLASSIFIER_ADDRESS = "http://159.8.222.7:8083/pd" # brainik80c RIP
CLASSIFIER_ADDRESS = constants.HYDRA_TG_CLASSIFIER_ADDRESS #"13.82.136.127:8083/hydra_tg"

def hydra_tg(image_arrary_or_url,thresholds=constants.hydra_tg_thresholds):
   # data = msgpack.dumps({"image": image_arrary_or_url})
    data = {"imageUrl": image_arrary_or_url,"thresholds":thresholds}
#    dumped_data = json.dumps({"imageUrl": image_arrary_or_url})
#    resp = requests.get(CLASSIFIER_ADDRESS, dumped_data)
    resp = requests.get(CLASSIFIER_ADDRESS, data)
    if  200 <= resp.status_code < 300:
        return json.loads(resp.content)
#        return msgpack.loads(resp.content)
    else:
        raise Exception('hydra_tg failed, abort', resp.content)
