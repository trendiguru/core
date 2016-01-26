__author__ = 'Nadav Paz'

import time
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import subprocess
import csv

import tldextract

import pymongo
from rq import Queue

from . import whitelist
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


def monitize():
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


def get_domain(url):
    if not isinstance(url, str):
        url = str(url)
    return tldextract.extract(url).registered_domain


def get_white_list():
    idx = 0
    for doc in db.images.find():
        domain = get_domain(doc['page_urls'][0])
        if not db.white_list.find_one({'domain.name': domain}):
            db.white_list.insert({'long_urls': [{'name': doc['page_urls'][0], 'count': 0}],
                                  'domain': {'name': domain, 'count': 0}})
        else:
            if not db.white_list.find_one({'long_url.name': doc['page_urls'][0]}):
                db.white_list.update_one({'domain.name': domain}, {'$inc': {'domain.count': 1},
                                                                   '$push': {'long_urls': {'name': doc['page_urls'][0],
                                                                                           'count': 0}}})
            else:
                db.white_list.update_one({'domain.name': domain}, {'$inc': {'domain.count': 1, 'long_url.count': 1}})
        if not idx % 100:
            print "performing {0}th doc".format(idx)
        idx += 1


def get_last_images(time_unit, num):
    if time_unit == 'day':
        curs = db.images.find({'saved_date': {'$gt': datetime.datetime.utcnow() - datetime.timedelta(days=num)}})
    elif time_unit == 'hour':
        curs = db.images.find({'saved_date': {'$gt': datetime.datetime.utcnow() - datetime.timedelta(hours=num)}})
    elif time_unit == 'minute':
        curs = db.images.find({'saved_date': {'$gt': datetime.datetime.utcnow() - datetime.timedelta(minutes=num)}})
    return curs


def get_top_viewed_images(x):
    dict = {}
    curs = db.images.find({'views': {'$gt': 1}}).sort([('views', pymongo.DESCENDING)]).limit(x)
    for doc in curs:
        dict[doc['image_urls'][0]] = doc['views']
    return dict


def check_relevancy():
    relevant_count = 0
    how_many_images = 0
    for doc in db.white_list.find():
        if doc['domain']['name']:
            splitted = doc['domain']['name'].split('/')[2]
            if splitted[:3] == 'www':
                splitted = splitted[4:]
            if splitted in whitelist.fullList:
                relevant_count += 1
                how_many_images += doc['domain']['count']
                print relevant_count
                print how_many_images
                print splitted
                time.sleep(0.3)


def download_last_x_logs(x):
    command = 'gsutil'
    page = 'gs://fzz_logs'
    address = "/home/developer/logs"
    os.chdir(address)
    if x == 'all':
        last_x_log = subprocess.check_output([command, 'ls ' + page]).split('\n')[:-1]
    else:
        last_x_log = subprocess.check_output([command, 'ls ' + page]).split('\n')[-(x + 1):-1]
    saved_logs = []
    for log in last_x_log:
        filename = log[len(page) + 1:] + ".csv"
        subprocess.call([command, 'cp ' + log + ' ' + filename])
        saved_logs.append(address + '/' + filename)
    return saved_logs


def save_log_to_mongo(log_file, delete_after=True, first_time=False):
    csv_file = open(log_file, 'r')
    reader = csv.DictReader(csv_file)
    docs_list = []

    for doc in reader:
        view = {'ip': doc['c_ip'], 'time': datetime.datetime.utcfromtimestamp((int(doc['time_micros'])) / 1e6)}
        page = {'url': doc['cs_referer'], 'view_count': 1, 'views': [view]}
        domain = get_domain(doc['cs_referer'])
        # if domain is already in the DB:
        if db.log.find_one({'domain': domain}):
            db.log.update_one({'domain': domain}, {'$addToSet': {'cs_uri': doc['cs_uri']},
                                                   '$inc': {'count': 1}})
            # if page is already in the DB:
            if db.log.find_one({'pages.url': doc['cs_referer']}):
                db.log.update_one({'pages.$.url': doc['cs_referer']}, {'$push': {'pages.$.views': view},
                                                                       '$inc': {'pages.$.view_count': 1}})
            # new page
            else:
                db.log.update_one({'domain': domain}, {'$push': {'pages': page}})
        # new domain
        else:
            docs_list.append({'domain': domain, 'count': 1, 'cs_uri': [doc['cs_uri']], 'pages': [page]})

    if len(docs_list):
        db.log.insert_many(docs_list)
    print "sum of all domains in db.log is {0}".format(db.log.count())
    print '\n'
    csv_file.close()
    if delete_after:
        os.remove(log_file)


def get_top_x(x):
    top = db.log.find({}).sort('count', pymongo.DESCENDING)
    print "Top " + str(x) + " sites:"
    for doc in top.limit(x):
        print "{0}: {1}".format(doc['domain'], doc['count'])


def run():
    logs = download_last_x_logs(1)
    for log in logs:
        save_log_to_mongo(log)
    get_top_x(100)


if __name__ == "__main__":
    run()