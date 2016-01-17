__author__ = 'liorsabag'

import rq
from rq.job import Job
from .constants import redis_conn


rq.push_connection(redis_conn)


def avg_processing_time_for_q(name, n=100):
    fjr = rq.registry.FinishedJobRegistry(name)
    last_n_finished_jobs = (Job.fetch(jid) for jid in fjr.get_job_ids(-n, -1))
    processing_times = [j.ended_at - j.started_at for j in last_n_finished_jobs if j.ended_at and j.started_at]
    return processing_times
    # avg = sum(processing_times) / float(len(processing_times))
    # print "Avg processing time: " + str(avg)
    # print len(processing_times)

# Time info gets saved on redis as string '2016-01-14T17:06:58Z'
# But we need microsecond resolution

