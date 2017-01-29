import requests
from bson import json_util as json
from .constants import db

SLACK_URL = "https://hooks.slack.com/services/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"
JOKE_URL = "http://api.icndb.com/jokes/random"

def notify_new_image(query):
    im = db.images.find_one(query, {"image_urls":1, "page_urls":1, "domain":1})
    message = "Also, there's a new <{0}|image> on <{1}|{2}>".format(im["image_urls"][-1], im["page_urls"][-1], im.get("domain", "page"))
    try:
        joke = requests.get(JOKE_URL).json()["value"]["joke"]
        message = "{0}\n{1}".format(joke, message)
    except:
        pass
    
    send_message("{0}\n{1}".format(joke, message))

def send_message(msg):
    payload = {"text": msg}
    resp = requests.post(SLACK_URL, data=json.dumps(payload))
