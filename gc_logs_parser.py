import os
import subprocess
import time
import datetime
import csv
import tldextract
from . import constants
db = constants.db
log_blacklist = ['googleapis.com', 'youtube.com']


def download_last_x_logs(x):
    """
    Downloads last X log-files
    :param x: if x='all' => downloads all logs
              else - download x last log files
    :return: list of strings - log files addresses
    """
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


def save_log_to_mongo(log_file, delete_after=True):
    start = time.time()
    print "starting to update the log.."
    csv_file = open(log_file, 'r')
    reader = csv.DictReader(csv_file)
    docs_list = []
    idx = 1
    for request in reader:
        view = {'ip': request['c_ip'], 'time': datetime.datetime.utcfromtimestamp((int(request['time_micros'])) / 1e6)}
        page = {'url': request['cs_referer'], 'view_count': 1, 'views': [view]}
        domain = get_domain(request['cs_referer'])
        # if page url valid to index
        if len(page['url']) < 1024 and domain not in log_blacklist:
            if db.log.update_one({'domain': domain}, {'$addToSet': {'cs_uri': request['cs_uri']},
                                                      '$inc': {'count': 1}}).acknowledged:
                if not db.log.update_one({'pages.url': page['url']}, {'$push': {'pages.$.views': view},
                                                                      '$inc': {'pages.$.view_count': 1}}).acknowledged:
                    # new page
                    db.log.update_one({'domain': domain}, {'$push': {'pages': page}})
            # new domain
            else:
                docs_list.append({'domain': domain, 'count': 1, 'cs_uri': [request['cs_uri']], 'pages': [page]})
        idx += 1
    if len(docs_list):
        db.log.insert_many(docs_list)
    print "done updating.. took {0} minutes".format((time.time() - start) / 60)
    print "sum of all domains in db.log is {0}".format(db.log.count())
    print '\n'
    csv_file.close()
    if delete_after:
        os.remove(log_file)


def get_domain(url):
    if not isinstance(url, str):
        url = str(url)
    return tldextract.extract(url).registered_domain