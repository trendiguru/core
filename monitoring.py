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


test_q = Queue('test_q', connection=redis_conn)
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


def run():
    checklist = {'redis_conn': {'flag': 0, 'start': 0},
                 'rq_functionality': {'flag': 0, 'start': 0},
                 'mongo_conn': {'flag': 0, 'start': 0},
                 'mongo_functionality': {'flag': 0, 'start': 0},
                 '0_inserts': {'flag': 0, 'start': 0}}
    while 1:
        print "starting a check in {0}".format(time.ctime())

        # REDIS & RQ

        # basic connection

        if not redis_conn.ping():
            if checklist['redis_conn']['flag'] and time.time() - checklist['redis_conn']['start'] < 3600:
                break
            stats = {'massege': 'FAILED TO CONNECT REDIS !', 'date': time.ctime()}
            email(stats, 'REDIS CONNECTION', [lior, nadav])
            checklist['redis_conn']['flag'] = 1
            checklist['redis_conn']['start'] = time.time()

        # putting on queue

        try:
            job = test_q.enqueue(return_1)
            time.sleep(0.1)
            if job.is_failed:
                if checklist['rq_functionality']['flag'] and time.time() - checklist['rq_functionality'][
                    'start'] < 3600:
                    break
                stats = {'massege': 'TEST JOB IS FAILED!', 'date': time.ctime()}
                email(stats, 'FAILED TO ENQUEUE', [lior, nadav])
                checklist['rq_functionality']['flag'] = 1
                checklist['rq_functionality']['start'] = time.time()
        except Exception as e:
            if checklist['rq_functionality']['flag'] and time.time() - checklist['rq_functionality']['start'] < 3600:
                break
            stats = {'massege': e.message, 'date': time.ctime()}
            email(stats, 'FAILED TO ENQUEUE', [lior, nadav])
            checklist['rq_functionality']['flag'] = 1
            checklist['rq_functionality']['start'] = time.time()

        # MONGO

        # basic connection

        try:
            db = pymongo.MongoClient(host=os.environ["MONGO_HOST"], port=int(os.environ["MONGO_PORT"])).mydb
        except Exception as e:
            if checklist['mongo_conn']['flag'] and time.time() - checklist['mongo_conn']['start'] < 3600:
                break
            stats = {'massege': e.message, 'date': time.ctime()}
            email(stats, 'FAILED TO CONNECT MONGO !', [lior, nadav])
            checklist['mongo_conn']['flag'] = 1
            checklist['mongo_conn']['start'] = time.time()

        # inserting a doc to db.test

        try:
            db.test.insert_one({'name': 'test'})
            db.test.delete_one({'name': 'test'})
        except Exception as e:
            if checklist['mongo_functionality']['flag'] and time.time() - checklist['mongo_functionality'][
                'start'] < 3600:
                break
            stats = {'massege': e.message, 'date': time.ctime()}
            email(stats, 'FAILED TO INSERT TO DB.TEST', [lior, nadav])
            checklist['mongo_functionality']['flag'] = 1
            checklist['mongo_functionality']['start'] = time.time()

        # IMAGES INSERTS

        images_init = db.images.count()
        time.sleep(60)
        if not db.images.count() - images_init:
            if checklist['0_inserts']['flag'] and time.time() - checklist['0_inserts']['start'] < 3600:
                break
            stats = {'massege': '0 IMAGES WERE INSERTED IN THE LAST MINUTE TO DB.IMAGES !!', 'date': time.ctime()}
            email(stats, '0 INSERTS', [lior, nadav])
            checklist['0_inserts']['flags'] = 1
            checklist['0_inserts']['start'] = time.time()

        for type_of, error in checklist.iteritems():
            if time.time() - error['start'] > 3600:
                checklist[type_of]['start'] = time.time()
                checklist[type_of]['flag'] = 0


if __name__ == "__main__":
    run()