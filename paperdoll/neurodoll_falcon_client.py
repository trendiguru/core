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
    urls = ['http://healthyceleb.com/wp-content/uploads/2014/03/Nargis-Fakhri-Main-Tera-Hero-Trailer-Launch.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg',
            'http://diamondfilms.com.au/wp-content/uploads/2014/08/Fashion-Photography-Sydney-1.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/02/main-1.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/02/1.-Strategic-Skin-Showing.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/02/3.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/02/4.jpg',
            'http://pinmakeuptips.com/wp-content/uploads/2016/03/Adding-Color-to-Your-Face.jpg',
            'http://images5.fanpop.com/image/photos/26400000/Cool-fashion-pics-fashion-pics-26422922-493-700.jpg',
            'http://allforfashiondesign.com/wp-content/uploads/2013/05/style-39.jpg',
            'http://s6.favim.com/orig/65/cool-fashion-girl-hair-Favim.com-569888.jpg',
            'http://s4.favim.com/orig/49/cool-fashion-girl-glasses-jeans-Favim.com-440515.jpg',
            'http://s5.favim.com/orig/54/america-blue-cool-fashion-Favim.com-525532.jpg',
            'http://favim.com/orig/201108/25/cool-fashion-girl-happiness-high-Favim.com-130013.jpg'
    ] #
    for u in urls+url:
        nd(u,get_combined_results=True)