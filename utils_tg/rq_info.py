
from redis import StrictRedis
from rq import push_connection, get_failed_queue, Queue
from rq.job import Job
import time
from trendi import constants
import json

con = constants.redis_conn

def queue_counts():
    #fq = rq.Queue("failed", connection=constants.redis_conn)
    all_queues = Queue.all(connection = con)
    count_dict = {}
    t = time.time()
    count_dict['time'] = t
    for queue in all_queues:
        a_queue = Queue(queue, connection = con)
        count = a_queue.count()
        count_dict[queue] = count
    return count_dict

def get_info_by_number(job_no):
    fq = Queue('failed', connection = con)
    count = fq.count
    print('FAILED JOBS COUNT:'+str(count))
    failed_jobs = fq.jobs
    for job in failed_jobs:
#        for key in dict:
 #           print(key,dict[key])
        if job.id == job_no:
            dict = job.to_dict()
            print('found job:')
            st = json.dumps(dict, sort_keys=True,indent=4, separators=(',', ': '))
            print st
            return dict
    print 'didnt find job '+str(job_no)
    return None

def failed_info():
    #fq = rq.Queue("failed", connection=constants.redis_conn)
    fq = Queue('failed', connection = con)
    count = fq.count
    print('FAILED JOBS COUNT:'+str(count))
    failed_jobs = fq.jobs
    reasons_dict = {}
    for job in failed_jobs:
        dict = job.to_dict()
#        for key in dict:
 #           print(key,dict[key])
        if 'exc_info' in dict:
            exception_info = dict['exc_info']
            if exception_info == '' or exception_info is None:
                exception_info = 'no info given\nno info given'
        else:
            exception_info = 'no info given\nno info given'
        lines = exception_info.split('\n')
        reason = lines[0]
        if len(lines)>1:
            reason = lines[-2]
        if 'NoSuchJobError' in reason or 'InvalidDocument' in reason or 'WiredTigerIndex' in reason:
#        if len(reason)>40:
            reason=reason[0:34]  #job # is different each time so avoid that
        if 'paperdoll failed on this file' in reason:
            reason=reason[0:42]  #job # is different each time so avoid that

 #       print('exc info:')
 #       print str(exception_info)
        print('reason: '+ str(reason))
 #       raw_input('enter to continue')
        if reason in reasons_dict:
            jobs_with_same_reason = reasons_dict[reason]
            jobs_with_same_reason.append(dict)
        else:
            reasons_dict[reason]=[dict]
#        for reason in reasons_dict:
 #           print reason, str(len(reasons_dict[reason]))
#    print reasons_dict

    return reasons_dict

def get_reasons():
    reasons_dict = failed_info()
    small_dict = {}
    timestring = time.strftime("%H:%M:%S %d/%m/%Y")
    small_dict['timestamp='+timestring] = 9999999
    for reason in reasons_dict:
        small_dict[reason] = len(reasons_dict[reason])
    sorted_dict = sorted(small_dict.items(),key=lambda x:x[1],reverse=True)

    return sorted_dict

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
    reasons = get_reasons()
    st = json.dumps(reasons, sort_keys=True,indent=4, separators=(',', ': '))
    print st
    with open('errordump.txt', 'a') as outfile:
        json.dump(reasons, outfile, sort_keys=True,indent=4, separators=(',', ': '))

    get_info_by_number('dc4ebb1f-9fbb-4ee6-ad5f-0f7e83f5e44a')


#    count_dict = queue_counts()
 #   print('counts: '+str(count_dict))
  #  j = json.dumps(count_dict)
  #  with open('queue_counts.json','a') as f:
   #     f.write(j)
    #time.sleep(10)
#
    #     reasons_dict = failed_info()
 #   print_reasons(reasons_dict)

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