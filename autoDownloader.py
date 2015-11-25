__author__ = 'yonatan'

from datetime import datetime

from rq_scheduler import Scheduler

from constants import redis_conn

scheduler = Scheduler(connection=redis_conn)


# remove old jobs
jobs = scheduler.get_jobs()
print jobs
for job in jobs:
    scheduler.cancel(job)

# set new jobs
scheduler.schedule(scheduled_time=datetime.now(),
                   func="trendi.caffeDocker.do",
                   repeat=5,
                   interval=20)
scheduler.enqueue_at(datetime(2015, 11, 25, 13, 39), "trendi.caffeDocker.do")
print ("new job list:")
jobs = scheduler.get_jobs()
print jobs
