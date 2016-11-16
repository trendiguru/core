from jaweson import msgpack
import requests

FEATURES =  {"collar":{"server":""},
             "sleeve":{"server":""},
             "dress_length":{"server":""},
             "style":{"server":""},
             "gender":{"server":""}}
         

def feature_url(feature):
    return FEATURES[feature]["server"].rstrip('/') + "/" + feature

for f in FEATURES:
    f["url"] = feature_url(f)

def get(feature, image_or_url):
    data = msgpack.dumps({"image_or_url": image_or_url})
    resp = requests.post(FEATURE[feature]["url"], data=data)
    return msgpack.loads(resp.content)


