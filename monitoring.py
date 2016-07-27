
import smtplib
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
    while not db.images.find_one({'image_urls': MINUTE_IMAGE_URL}):
        time.sleep(1)
    db.images.delete_one({'image_urls': MINUTE_IMAGE_URL})


def log_failed():
    while failed.count():
        job = failed.dequeue()
        print "failed at {0} ".format(str(job.created_at))
        print "job origin: {0}".format(job.to_dict()['origin'])
        print "execution info: {0}".format(job.to_dict()['exc_info'])
        print '\n'

if __name__ == "__main__":
    dummy_image()
    log_failed()
