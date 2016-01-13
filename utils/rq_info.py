
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
    reasons_dict = {}
    for job in failed_jobs:
        dict = job.to_dict()
 #       for key in dict:
#            print(key,dict[key])
        if 'exc_info' in dict:
            exception_info = dict['exc_info']
        else exception_info = 'no info given\nno info given'
        lines = exception_info.split('\n')
        reason = lines[-2]
        if reason in reasons_dict:
            jobs_with_same_reason = reasons_dict[reason]
            jobs_with_same_reason.append(dict)
        else:
            reasons_dict[reason]=[dict]

def print_reasons(reasons_dict):
    for reason in reasons_dict:
        arr = reasons_dict[reason]
        n=len(arr)
        print('reason:{0}, n:{1}'.format(reason,n))
#        created_time = dict['created_at']
 #       exception_info = dict['exc_info']
  #      #print('exception info:'+str(exception_info))
   #     lines = exception_info.split('\n')
#        enqueue_time = dict['enqueued_at']
 #       end_time = dict['ended_at']
  #      func = ''
  #      if 'func_name' in dict:
   #        func = dict['func_name']
    #    args = dict['args']
     #   id = dict['id']
      #  print('reason {5} id {0} created {1} ended {2} function {3} args {4}'.format(id,created_time,end_time,func,args,reason))


if __name__ == "__main__":
    reasons_dict = failed_info()
    print_reasons(reasons_dict)

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