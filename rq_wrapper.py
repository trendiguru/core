__author__ = 'liorsabag'

from rq.job import Job
from rq import Queue

q = Queue()


def enqueue_after_many(self, list_of_unqueued_jobs, dependent_job):
    remaining_dependencies = len(list_of_unqueued_jobs)
    def wrap_func(func):
        def new_func(*args):
            # run func(args) then let everyone know
            result = func(*args)
            enqueued_dep_job = dependency_finished(result)
            return result, enqueued_dep_job
        return new_func

    def dependency_finished(result):
        if remaining_dependencies > 0:
            remaining_dependencies -= 1
        else:
            return self.enqueue_job(dependent_job)

        pass

    for u_job in list_of_unqueued_jobs:
        u_job.func = wrap_func(u_job.func)
    enqueued_jobs = [self.enqueue_job(u_job) for u_job in list_of_unqueued_jobs]


# consider creating the future job and adding dependencies as meta


