__author__ = 'Nadav Paz'

import time

from .constants import db


def calibrate_time():
    for doc in db.monitoring.find():
        db.monitoring.update_one({'queue': doc['queue']}, {'$set': {'start': time.time()}})


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