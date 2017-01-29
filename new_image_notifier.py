import requests
from bson import json_util as json
from .constants import db

SLACK_URL = "https://hooks.slack.com/services/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"

def notify_new_image(query):
    im = db.images.find_one(query, {"image_urls":1, "page_urls":1})
    message = "New <{0}|image> processed on page <{1}|page>".format(im["image_urls"][-1], im["page_urls"][-1])
    send_message(message)

def send_message(msg):
    payload = {"text": msg}
    resp = requests.post(SLACK_URL, data=json.dumps(payload))
