__author__ = 'Nadav Paz'

import time

from rq import Queue

from .constants import redis_conn

from .constants import db
from . import paperdolls


def reset_time():
    for doc in db.monitoring.find():
        db.monitoring.update_one({'queue': doc['queue']}, {'$set': {'start': time.time()}})


def reset_amount():
    for doc in db.monitoring.find():
        db.monitoring.update_one({'queue': doc['queue']}, {'$set': {'count': 0}})


def check_queues():
    print "find_similar queue: {0}".format(paperdolls.q1.count)
    print "find_top_n queue: {0}".format(paperdolls.q2.count)
    print "find_similar queue: {0}".format(Queue('failed', connection=redis_conn).count)

    # def get_minutely():
    # how much time back should I save?
    # what exactly to save?
    # which stats should I get out of the stats and how?

    
        # # TIME STATS
        # def moment_information():
        # def minute_information():
        # def hour_information():
        # def daily_information():
        # def weekly_information():
        #
        # # OTHER STATS
        # def queues_status():
        # def workers_status():
        # def
        #
        # if __name__ == '__main__':
        # print "starting monitoring.."