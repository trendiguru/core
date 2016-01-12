
from redis import StrictRedis
from rq import push_connection, get_failed_queue, Queue
from rq.job import Job


con = StrictRedis()
push_connection(con)

def div_by_zero(x):
    return x / 0

job = Job.create(func=div_by_zero, args=(1, 2, 3))
job.origin = 'fake'
job.save()
fq = get_failed_queue()
fq.quarantine(job, Exception('Some fake error'))
assert(fq.count == 1)

fq.requeue(job.id)

assert(fq.count == 0)
assert(Queue('fake').count == 1)