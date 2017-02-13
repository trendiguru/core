__author__ = 'jeremy'

from jaweson import msgpack
import requests

#
CLASSIFIER_ADDRESS = "http://13.82.136.127:8081/hydra"
#docker-user@13.82.136.127
#thats allison

def bring_forth_the_hydra(image_array_or_url, gpu=1):
    params = {}
    if gpu:
        params['gpu'] = gpu
    if params == {}:
        params = None #not sure if this is necesary but the original line (below) made it happen
        #params = params={"categoryIndex": category_index} if category_index else None
    print('params coming into hydra:'+str(params))
    data = msgpack.dumps({"image": image_array_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data, params=params)
    return msgpack.loads(resp.content)

if __name__ == "__main__":
    url = 'http://www.thecantoncitizen.com/wp-content/uploads/2013/08/dunkin-robbery.jpg'
    bring_forth_the_hydra(url)