
from redis import StrictRedis
from rq import push_connection, get_failed_queue, Queue
from rq.job import Job

from trendi import constants

con = constants.redis_conn


def failed_info():
    #fq = rq.Queue("failed", connection=constants.redis_conn)

    fq = Queue('failed', connection = con)
    count = fq.count
    print('count:'+str(count))

    failed_jobs = fq.jobs
    for job in failed_jobs:
        exception_info = job.exc_info
        print('exception info:'+str(exception_info[0:60]))


if __name__ == "__main__":
    failed_info()