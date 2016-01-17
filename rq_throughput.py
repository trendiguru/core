__author__ = 'liorsabag'

import logging
import time
import rq
from rq.job import Job
from .constants import redis_conn

rq.push_connection(redis_conn)


def avg_processing_time_for_q(name, n=100):
    fjr = rq.registry.FinishedJobRegistry(name)
    last_n_finished_jobs = (Job.fetch(jid) for jid in fjr.get_job_ids(-n, -1))
    processing_times = [(j.ended_at - j.started_at).total_seconds()
                        for j in last_n_finished_jobs if j.ended_at and j.started_at]
    lpt = len(processing_times)
    if lpt != n:
        logging.warn("Not all jobs have an 'ended_at' property, only {0} of {1}".format(lpt, n))
    return sum(processing_times)/float(lpt)


def monitor_q_throughput(name, n=100, interval=1):
    while True:
        avg = avg_processing_time_for_q(name, n)
        print "Avg of last {0} processing times: {1}".format(n, avg)
        time.sleep(interval)


