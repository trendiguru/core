from jaweson import msgpack
import requests

#
CLASSIFIER_ADDRESS = "http://169.45.147.210:8080/mnc"
#thats brainim60


def mnc(image_array_or_url,cat_to_look_for='person'):
    params = {}
    if cat_to_look_for:
        params['catToLookFor'] = cat_to_look_for
    if params == {}:
        params = None #not sure if this is necesary but the original line (below) made it happen
        #params = params={"categoryIndex": category_index} if category_index else None
    print('params coming into pd:'+str(params))
    data = msgpack.dumps({"image": image_array_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data, params=params)
    return msgpack.loads(resp.content)
    
