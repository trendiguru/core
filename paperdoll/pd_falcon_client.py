__author__ = 'liorsabag'

from jaweson import msgpack
import requests

#
# CLASSIFIER_ADDRESS = "http://37.58.101.170:8080/pd"  # Braini5
# CLASSIFIER_ADDRESS = "http://37.58.101.173:8082/pd"  # Braini2
# CLASSIFIER_ADDRESS = "http://159.8.222.7:8083/pd" # brainik80c RIP
CLASSIFIER_ADDRESS = "http://acs-1agents.westeurope.cloudapp.azure.com/pd"


def pd(image_arrary_or_url):
    data = msgpack.dumps({"image": image_arrary_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data)
    if  200 <= resp.status_code < 300:
        return msgpack.loads(resp.content)
    else:
        raise Exception('PD FAILED', resp.content)
