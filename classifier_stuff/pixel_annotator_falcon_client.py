
#this is doing a post, which the javascript is doing , so I don't think
#we need this file at all...


from jaweson import msgpack
import requests

#
CLASSIFIER_ADDRESS = "http://169.45.147.210:8083/pixlevel_annotator"
#thats brainim60

    params = {}
    if params == {}:
        params = None #not sure if this is necesary but the original line (below) made it happen
        #params = params={"categoryIndex": category_index} if category_index else None
    print('params coming into mlb:'+str(params))
    #print('image coming into mlb:'+str(image_array_or_url))
    data = msgpack.dumps({"image": image_array_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data, params=params)
    return msgpack.loads(resp.content)
    
