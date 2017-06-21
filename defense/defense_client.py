from jaweson import msgpack
import requests

from trendi import constants

CLASSIFIER_ADDRESS = constants.FRCNN_CLASSIFIER_ADDRESS # "http://13.82.136.127:8084/hls"

def detect(img_arr, roi=[]):
    print('using addr '+str(CLASSIFIER_ADDRESS))
    data = {"image": img_arr}
    if roi:
        print "Make sure roi is a list in this order [x1, y1, x2, y2]"
        data["roi"] = roi
    serialized_data = msgpack.dumps(data)
    resp = requests.post(CLASSIFIER_ADDRESS, data=serialized_data)
    return msgpack.loads(resp.content)


YOLO_HLS_ADDRESS = constants.YOLO_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8082/hls"
#http://13.82.136.127:8082/hls?imageUrl=http://videos.cctvcamerapros.com/images/installs/gas-station-surveillance/600/4-camera-surveillance-system.jpg&net=pyyolo
def detect_hls(img_arr, roi=[]):
    print('using addr '+str(YOLO_HLS_ADDRESS))
    data = {"image": img_arr}
    if roi:
        print "Make sure roi is a list in this order [x1, y1, x2, y2]"
        data["roi"] = roi
    serialized_data = msgpack.dumps(data)
    resp = requests.post(YOLO_HLS_ADDRESS, data=serialized_data)
    return msgpack.loads(resp.content)

if __name__=="__main__":
    url = 'http://videos.cctvcamerapros.com/images/installs/gas-station-surveillance/600/4-camera-surveillance-system.jpg'
    res = detect_hls(url)
    print('result for {}:\n{}'.format(url,res))