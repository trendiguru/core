
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
    print('len:'+str(len(failed_jobs)))
    for job in failed_jobs:
        dict = job.to_dict()
        for key in dict:
            print(key,dict[key])
        created_time = dict['created_at']
        exception_info = dict['exc_info']
        #print('exception info:'+str(exception_info))
        lines = exception_info.split('\n')
        enqueue_time = dict['enqueued_at']
        end_time = dict['ended_at']
        func =dict['function_name']
        args = dict['args']
        id = dict['id']
        print('id {0} created {1} ended {2} function {3} args {4}'.format(id,created_time,end_time,func,args))

if __name__ == "__main__":
    failed_info()

'''
    job.args                 job.dependency           job.fetch                job.id                   job.kwargs               job.return_value
job.cancel               job.dependents_key       job.func                 job.instance             job.meta                 job.save
job.cleanup              job.dependents_key_for   job.func_name            job.is_failed            job.origin               job.set_id
job.connection           job.description          job.get_call_string      job.is_finished          job.perform              job.set_status
job.create               job.ended_at             job.get_id               job.is_queued            job.refresh              job.status
job.created_at           job.enqueued_at          job.get_result_ttl       job.is_started           job.register_dependency  job.timeout
job.data                 job.exc_info             job.get_status           job.key                  job.result               job.to_dict
job.delete               job.exists               job.get_ttl              job.key_for              job.result_ttl           job.ttl
'''