from jaweson import msgpack
import requests

#
CLASSIFIER_ADDRESS = "http://37.58.101.173:8080/nd"


def pd(image_array_or_url, category_index=None):
    params = None
    if category_index:
        params['categoryIndex'] = category_index
    if get:
        params['getMultilabelResults'] = get_multilabel_results
    print('params coming into pd:'+str(params))
        #params = params={"categoryIndex": category_index} if category_index else None
    data = msgpack.dumps({"image": image_array_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data, params=params)
    return msgpack.loads(resp.content)
    
