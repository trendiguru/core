
from redis import StrictRedis
from rq import push_connection, get_failed_queue, Queue
from rq.job import Job

from trendi import constants

con = constants.redis_conn
#con = StrictRedis()
push_connection(con)

def failed_info():
    fq = get_failed_queue()
    count = fq.count()
    print('count:'+str(count))
#    fq.requeue(job.id)

if __name__ == "__main__":
    failed_info()