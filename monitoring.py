__author__ = 'Nadav Paz'

import time
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pymongo

from rq import Queue

from .constants import db
from .constants import redis_conn


test_q = Queue('test', connection=redis_conn)
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


def return_1():
    return 1


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


def run():
    while 1:

        # REDIS & RQ

        # basic connection

        if not redis_conn.ping():
            stats = {'massage': 'FAILED TO CONNECT REDIS !', 'date': time.ctime()}
            email(stats, 'REDIS CONNECTION', [lior, nadav])

        # putting on queue

        try:
            job = test_q.enqueue(return_1)
            print "Did job start? : {0}".format(job.is_started)
            time.sleep(0.01)
            if job.is_failed:
                stats = {'massage': 'TEST JOB IS FAILED!', 'date': time.ctime()}
                email(stats, 'FAILED TO ENQUEUE', [lior, nadav])
        except Exception as e:
            stats = {'massage': e.message, 'date': time.ctime()}
            email(stats, 'FAILED TO ENQUEUE', [lior, nadav])

        # MONGO

        # basic connection

        try:
            db = pymongo.MongoClient(host=os.environ["MONGO_HOST"], port=int(os.environ["MONGO_PORT"])).mydb
        except Exception as e:
            stats = {'massage': e.message, 'date': time.ctime()}
            email(stats, 'FAILED TO CONNECT MONGO !', [lior, nadav])

        # inserting a doc to db.test

        try:
            db.test.insert_one({'name': 'test'})
            db.delete.insert_one({'name': 'test'})
        except Exception as e:
            stats = {'massage': e.message, 'date': time.ctime()}
            email(stats, 'FAILED TO INSERT TO DB.TEST', [lior, nadav])

        time.sleep(10)


if __name__ == "__main__":
    run()