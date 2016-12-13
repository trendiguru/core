from jaweson import msgpack
import requests

#
#CLASSIFIER_ADDRESS = "http://37.58.101.173:8080/nd" #braini2
CLASSIFIER_ADDRESS = "http://13.69.27.202:8080/nd" #azure gpu-1

#run gunicorn -b :8080 --workers 4 --timeout 300 trendi.paperdoll.neurodoll_falcon:api


def pd(image_array_or_url, category_index=None,get_multilabel_results=None,get_combined_results=None,get_layer_output=None,get_all_graylevels=None,threshold=None):
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
    
