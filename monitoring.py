import traceback
import fluent.event
import fluent.sender
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import time
from bson import json_util
from rq import Queue
from .constants import redis_conn, db


failed = Queue('failed', connection=redis_conn)
NADAV = 'nadav@trendiguru.com'
LIOR = 'lior@trendiguru.com'
KYLE = 'kyle@trendiguru.com'
JEREMY = 'jeremy@trendiguru.com'
YONTI = 'yontilevin@gmail.com'
SENDER = 'Notifier@trendiguru.com'
all = 'members@trendiguru.com'
API_URL = 'http://api.trendi.guru/images'
MINUTE_IMAGE_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/James_Corden_at_2015_PaleyFest.jpg/800px-James_Corden_at_2015_PaleyFest.jpg'

fluent.sender.setup('google-fluentd', host='localhost', port=24224)


def email(stats, title, recipients):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = title
    msg['From'] = sender
    msg['To'] = ", ".join(recipients)

    txt = '<h3> date:\t' + stats['date'] + '</h3>\n<h3>' + \
          'massege:\t' + stats['massege'] + '</h3>\n'

    html = """\
    <html>
    <head>
    <style>
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    th, td {
        padding: 5px;
    }
    </style>
    </head>
    <body>"""
    html += txt

    part1 = MIMEText(html, 'html')
    msg.attach(part1)
    server = smtplib.SMTP('smtp-relay.gmail.com', 587)
    server.starttls()
    server.sendmail(sender, recipients, msg.as_string())
    server.quit()


def dummy_image():
    data = {"pageUrl": "dummy", "imageList": [MINUTE_IMAGE_URL]}
    requests.post(API_URL, data=json_util.dumps(data))
    image_obj = db.images.find_one({'image_urls': MINUTE_IMAGE_URL})
    start = time.time()
    while not image_obj:
        time.sleep(1)
        image_obj = db.images.find_one({'image_urls': MINUTE_IMAGE_URL})
        if time.time() - start > 60:
            raise RuntimeError("Can't find image in DB after 1 minute!")
    res_coll = image_obj['people'][0]['items'][0]['similar_results'].keys()[0]
    sim_result = image_obj['people'][0]['items'][0]['similar_results'][res_coll][0]
    prod_coll = res_coll + '_' + image_obj['people'][0]['gender']
    real_result = db[prod_coll].find_one({'id': sim_result['id']})
    uri = 'http://links.trendi.guru/tr/test/' + prod_coll + '/' + str(real_result['_id']) + '?ver=0.1&userId=1520444741.1451463705&winWidth=0&winHeight=0&rv=791214116&event=Result%20Clicked&overlay=roundDress&refererDomain=fashionseoul.com&PID=fashionseoul&publisherDomain=fashionseoul.com'
    requests.get(uri)
    db.images.delete_one({'image_urls': MINUTE_IMAGE_URL})


def log_failed():
    while failed.count:
        job = failed.dequeue()
        data = {'created_at': str(job.created_at),
                'origin': job.to_dict()['origin'],
                'traceback': job.to_dict()['exc_info']}
        report(data)


def report(ex):
    data = {'message': str(ex), 'serviceContext': {'service': 'google-fluentd'}}
    fluent.event.Event('errors', data)


if __name__ == "__main__":
    dummy_image()
    log_failed()
