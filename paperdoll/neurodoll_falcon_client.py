from jaweson import msgpack
import requests

#run gunicorn -b :8080 --workers 4 --timeout 300 trendi.paperdoll.neurodoll_falcon:api

from trendi import constants
CLASSIFIER_ADDRESS = constants.NEURODOLL_CLASSIFIER_ADDRESS #"http://acs-1agents.westeurope.cloudapp.azure.com/pd"




def nd(image_array_or_url, category_index=None,get_multilabel_results=None,get_combined_results=None,get_layer_output=None,get_all_graylevels=None,threshold=None):
    params = {}
    if category_index:
        params['categoryIndex'] = category_index
    if get_multilabel_results:
        params['getMultilabelResults'] = get_multilabel_results
    if get_combined_results:
        params['getCombinedResults'] = get_combined_results
    if get_layer_output:
        params['getLayerOutput'] = get_layer_output
    if get_all_graylevels:
        params['getAllGrayLevels'] = get_all_graylevels
    if threshold:
        params['threshold'] = threshold

#    if get_yolo:
#        params['getYolo'] = get_yolo
    if params == {}:
        params = None #not sure if this is necesary but the original line (below) made it happen
        #params = params={"categoryIndex": category_index} if category_index else None
    print('params coming into neurodoll falcon client:'+str(params))
    data = msgpack.dumps({"image": image_array_or_url})
    resp = requests.post(CLASSIFIER_ADDRESS, data=data, params=params)
    return msgpack.loads(resp.content)
    
if __name__ == "__main__":
    url = 'https://s-media-cache-ak0.pinimg.com/736x/ae/d7/24/aed7241fcb27ad888cabefb82696b553.jpg'
    nd(url)