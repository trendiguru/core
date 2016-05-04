__author__ = 'jeremy'
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        unicode_literals)

import logging
import sys
import time
from rq.compat import as_text, string_types, text_type

from rq.job import Job, JobStatus
from rq.registry import FinishedJobRegistry, StartedJobRegistry, clean_registries
from rq.utils import (ensure_list, enum, import_attribute, make_colorizer,
                    utcformat, utcnow, utcparse)

from rq.worker import Worker

import caffe
logging.basicConfig(level=logging.DEBUG)


green = make_colorizer('darkgreen')
yellow = make_colorizer('darkyellow')
blue = make_colorizer('darkblue')


class TgNNDoormanWorker(Worker):

    def main_work_horse(self, *args, **kwargs):
        raise NotImplementedError("Test worker does not implement this method")

    def execute_job(self, *args, **kwargs):
        """Execute job in same thread/process, do not fork()"""
        logging.debug('executing from tgworker')
        DEFAULT_WORKER_TTL = 1000
        DEFAULT_RESULT_TTL = 1000
        logger = logging.getLogger(__name__)
        logging.debug('checking to start engine in ej')
        caffe.set_mode_gpu();
        caffe.set_device(0);
        if not hasattr(self,'caffe_net'):
            prototxt = '/home/jeremy/caffenets/voc-fcn8s/solver.prototxt'
            caffemodel = '/home/jeremy/caffenets/voc-fcn8s/train_iter_457644.caffemodel'
            logging.debug('tring to get new caffe net')
            caffenet = caffe.Net(prototxt, caffemodel, caffe.TEST)
#            eng = matlab.engine.start_matlab()
#            engine_name = eng.engineName
            logging.debug('new caffe net obtained')
        else:
            logger.info('found extant caffemodel in ej')

        return self.perform_job(*args, **kwargs)

    def perform_job(self, job):
        """Performs the actual work of a job.  Will/should only be called
        inside the work horse's process.
        """
        self.prepare_job_execution(job)

        with self.connection._pipeline() as pipeline:
            started_job_registry = StartedJobRegistry(job.origin, self.connection)

            try:
                logging.debug('perform_job')
#                job.matlab_engine = self.matlab_engine
                job.caffenet = self.caffebet   #not sure this is needed
           #     logging.debug('pj engine:'+str(self.matlab_engine))
                logging.debug('pj args,kwargs:'+str(job._args)+','+str(job._kwargs))
                if len(job._args) > 0:  #got regular args not kwargs
                    new_args = (self.caffenet,)+job._args
                    logging.debug('tg pj new args:'+str(new_args))
                    job._args = new_args
                elif len(job._kwargs) > 0:  #got kwargs
                    job._kwargs['caffenet']=self.caffenet
                    logging.debug('tg/pj new kwargs:'+str(job._kwargs))
                with self.death_penalty_class(job.timeout or self.queue_class.DEFAULT_TIMEOUT):
                    rv = job.perform()
        # Pickle the result in the same try-except block since we need
        # to use the same exc handling when pickling fails
                job._result = rv

                self.set_current_job_id(None, pipeline=pipeline)

                result_ttl = job.get_result_ttl(self.default_result_ttl)
                if result_ttl != 0:
                    job.ended_at = utcnow()
                    job._status = JobStatus.FINISHED
                    job.save(pipeline=pipeline)

                    finished_job_registry = FinishedJobRegistry(job.origin, self.connection)
                    finished_job_registry.add(job, result_ttl, pipeline)

                job.cleanup(result_ttl, pipeline=pipeline)
                started_job_registry.remove(job, pipeline=pipeline)

                pipeline.execute()

            except Exception:
                job.set_status(JobStatus.FAILED, pipeline=pipeline)
                started_job_registry.remove(job, pipeline=pipeline)
                try:
                    pipeline.execute()
                except Exception:
                    pass
                self.handle_exception(job, *sys.exc_info())
                return False

        if rv is None:
            self.log.info('Job OK')
        else:
            self.log.info('Job OK, result = {0!r}'.format(yellow(text_type(rv))))

        if result_ttl == 0:
            self.log.info('Result discarded immediately')
        elif result_ttl > 0:
            self.log.info('Result is kept for {0} seconds'.format(result_ttl))
        else:
            self.log.warning('Result will never expire, clean up result key manually')

        return True
