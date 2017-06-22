import time
from collections import namedtuple
import cv2
import numpy as np
import requests
from jaweson import msgpack


class Image(object):
    def __init__(self, url=None, data_uri=None, arr=None):
        self._url = url
        self._data_uri = data_uri
        self._encoded_string = None
        self._encoded_arr = None
        self._arr = arr

    @property
    def arr(self):
        if self._arr:
            return self._arr
        else:
            self._arr = cv2.imdecode(self.encoded_arr, cv2.IMREAD_COLOR)
            return self._arr

    @property
    def encoded_arr(self):
        if self._encoded_arr:
            return self._encoded_arr
        # TODO: which of the following is faster on average?
        elif self._encoded_string:
            self._encoded_arr = np.fromstring(self._encoded_string, np.uint8)
            return self._encoded_arr
        elif self._arr:
            f, enc_jpg_np = cv2.imencode('.jpg', self._arr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            self._encoded_arr = enc_jpg_np
            return self._encoded_arr
        else:
            self._encoded_arr = np.fromstring(self.encoded_string, np.uint8)
            return self._encoded_arr

    @property
    def encoded_string(self):
        if self._encoded_string:
            return self._encoded_string
        else:
            if self._data_uri:
                encoded_data = self._data_uri.split(',')[1]
                self._encoded_string = encoded_data.decode('base64')
                return self._encoded_string
            elif self._url:
                response = requests.get(self._url, timeout=0.5)
                self._encoded_string = response.content
            else:
                raise IOError("No url or data_uri provided")

    @property
    def data_uri(self):
        if self._data_uri:
            return self._data_uri
        else:
            raise NotImplementedError("Don't yet know how to generate data uris")
            # return "data:image/jpeg;base64,{}".format(self._encoded_string.encode('base64'))



def url_to_np_array(url):
    response = requests.get(url, timeout=0.5)
    raw_np = np.fromstring(response.content, np.uint8)
    return cv2.imdecode(raw_np, cv2.IMREAD_COLOR)





def serialize(arr):
    f, enc_np = cv2.imencode('.png', arr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return msgpack.dumps(enc_np)


def deserialize(msg):
    enc_np = msgpack.loads(msg)
    return cv2.imdecode(enc_np, cv2.IMREAD_COLOR)


arr_cv2 = url_to_np_array("http://fazz.co/src/img/demo/gettyimages-492504614.jpg")
f, enc_jpg_np = cv2.imencode('.jpg', arr_cv2, [cv2.IMWRITE_JPEG_QUALITY, 100])
f, enc_png_np = cv2.imencode('.png', arr_cv2, [cv2.IMWRITE_PNG_COMPRESSION, 9])



start = time.time()
msg = msgpack.dumps(arr_cv2)
arr = msgpack.loads(msg)
print time.time() - start

start = time.time()
msg = msgpack.dumps(cv2.imencode('.png', arr_cv2, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1]);
arr = cv2.imdecode(msgpack.loads(msg), cv2.IMREAD_COLOR);
print time.time() - start

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

data_uri = "data:image/jpeg;base64,/9j/4AAQ..."
img = data_uri_to_cv2_img(data_uri)
cv2.imshow(img)