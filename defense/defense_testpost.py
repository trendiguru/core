__author__ = 'jeremy'

from jaweson import msgpack
import requests
import json
#
FRCNN_CLASSIFIER_ADDRESS = "http://13.82.136.127:8082/hls"
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
#    data = msgpack.dumps({"image": image_array_or_url})
    data = {"image": image_array_or_url}
    resp = requests.post(CLASSIFIER_ADDRESS, data=data, params=params)
    return msgpack.loads(resp.content)

def secure_the_homeland(image_array_or_url, gpu=1):
    params = {}
    if gpu:
        params['gpu'] = gpu
    if params == {}:
        params = None #not sure if this is necesary but the original line (below) made it happen
        #params = params={"categoryIndex": category_index} if category_index else None
    print('params coming into hls:'+str(params))


    #try POST
 # #   data = msgpack.dumps({"image": image_array_or_url})
 #    data_dict = {"image": image_array_or_url}
 #    dumped_data = json.dumps(data_dict)
 #    print('secure_the_homeland looking for a response to POST from '+str(FRCNN_CLASSIFIER_ADDRESS))
 #    print('data: '+str(data_dict))
 #    resp = requests.post(FRCNN_CLASSIFIER_ADDRESS, data=dumped_data)
 #    print('response  to POST:'+str(resp.content))
 #    print resp.content

    #try GET json dumps
    data_dict = {"imageUrl": image_array_or_url}
    dumped_data = json.dumps(data_dict)
    print('secure_the_homeland looking for a response to GET from '+str(FRCNN_CLASSIFIER_ADDRESS))
    print('params: '+str(data_dict))
    resp = requests.get(FRCNN_CLASSIFIER_ADDRESS,params=dumped_data)
    print('response  to GET:'+str(resp.content))
    print resp.content

    #try POST msgpack
 #
    # data_dict = {"image": image_array_or_url}
    # dumped_data = msgpack.dumps(data_dict)
    # print('secure_the_homeland looking for a response to POST from '+str(FRCNN_CLASSIFIER_ADDRESS))
    # print('data: '+str(data_dict))
    # resp = requests.post(FRCNN_CLASSIFIER_ADDRESS, data=dumped_data)
    # print('response  to POST:'+str(resp.content))
    # print resp.content

    #try GET msgpack
    data_dict = {"imageUrl": image_array_or_url}
    dumped_data = msgpack.dumps(data_dict)
    print('secure_the_homeland looking for a response to GET from '+str(FRCNN_CLASSIFIER_ADDRESS))
    print('params: '+str(data_dict))
    resp = requests.get(FRCNN_CLASSIFIER_ADDRESS,params=dumped_data)
    print('response  to GET:'+str(resp.content))
    print resp.content


if __name__ == "__main__":
    url = 'http://www.thecantoncitizen.com/wp-content/uploads/2013/08/dunkin-robbery.jpg'
    url = 'http://mediaweb.wftv.com/photo/2016/12/02/MTD-HOME%20BURGLARIES%20-%205pm%20-.jpgo_20161202234727517_6738720_ver1.0_640_360.jpg'
    url = 'https://cbsnewyork.files.wordpress.com/2011/12/surveillance-footage-of-suspect-in-shooting-of-nypd-officers-in-queens-12-2-11.jpg?w=420'
    resp = secure_the_homeland(url)
    #resp = bring_forth_the_hydra(url)
    print(resp)