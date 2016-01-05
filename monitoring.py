__author__ = 'Nadav Paz'

import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from rq import Queue

from .constants import db
from .constants import redis_conn


nadav = 'nadav@trendiguru.com'
lior = 'lior@trendiguru.com'
kyle = 'kyle@trendiguru.com'
jeremy = 'jeremy@trendiguru.com'
yonti = 'yontilevin@gmail.com'
sender = 'Notifier@trendiguru.com'
all = 'members@trendiguru.com'


def reset_time():
    for doc in db.monitoring.find():
        db.monitoring.update_one({'queue': doc['queue']}, {'$set': {'start': time.time()}})


def reset_amount():
    for doc in db.monitoring.find():
        db.monitoring.update_one({'queue': doc['queue']}, {'$set': {'count': 0}})


def check_queues():
    data = {'new_images': Queue('new_images', connection=redis_conn).count,
            'pd': Queue('pd', connection=redis_conn).count,
            'find_similar': Queue('find_similar', connection=redis_conn).count,
            'find_top_n': Queue('find_top_n', connection=redis_conn).count}


def email(stats, title, recipients):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = title
    msg['From'] = sender
    msg['To'] = ", ".join(recipients)

    txt = '<h3> date:\t' + stats['date'] + '</h3>\n<h3>' + \
          'massage:\t' + stats['massage'] + '</h3>\n'

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


if __name__ == "__main__":
    while 1:
        time.sleep(5)

        # REDIS & RQ

        # basic connection

        if not redis_conn.ping():
            stats = {'massage': 'No REDIS connection !', 'date': time.ctime()}
            email(stats, 'REDIS CONNECTION', [lior, nadav])
        else:
            stats = {'massage': 'REDIS connection is jussst fiiine !', 'date': time.ctime()}
            email(stats, 'REDIS CONNECTION', [lior, nadav])

            #
