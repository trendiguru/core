# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        unicode_literals)

import errno
import logging
import os
import random
import signal
import socket
import sys
import time
import traceback
import warnings
from datetime import timedelta
import os

from rq.compat import as_text, string_types, text_type

#ON braini1THIS GOES LIKE
from rq.connections import get_current_connection
from rq.exceptions import DequeueTimeout
from rq.job import Job, JobStatus
from rq.logutils import setup_loghandlers
from rq.queue import Queue, get_failed_queue
from rq.registry import FinishedJobRegistry, StartedJobRegistry, clean_registries
from rq.suspension import is_suspended
from rq.timeouts import UnixSignalDeathPenalty
from rq.utils import (ensure_list, enum, import_attribute, make_colorizer,
                    utcformat, utcnow, utcparse)
from rq.version import VERSION

from rq.worker import Worker

#ON PP-@ THIS GOES LIKE
#from .connections import get_current_connection
#from .exceptions import DequeueTimeout
#from .job import Job, JobStatus
#from .logutils import setup_loghandlers
#from .queue import Queue, get_failed_queue
#from .registry import FinishedJobRegistry, StartedJobRegistry, clean_registries
#from .suspension import is_suspended
#from .timeouts import UnixSignalDeathPenalty
#from .utils_tg import (ensure_list, enum, import_attribute, make_colorizer,
#                    utcformat, utcnow, utcparse)
#from .version import VERSION

#from .worker import Worker

import matlab.engine
logging.basicConfig(level=logging.DEBUG)


green = make_colorizer('darkgreen')
yellow = make_colorizer('darkyellow')
blue = make_colorizer('darkblue')


class TgWorker(Worker):

    def main_work_horse(self, *args, **kwargs):
        raise NotImplementedError("Test worker does not implement this method")

    def execute_job(self, *args, **kwargs):
        """Execute job in same thread/process, do not fork()"""
        logging.debug('executing from tgworker')
        DEFAULT_WORKER_TTL = 1000
        DEFAULT_RESULT_TTL = 1000
        logger = logging.getLogger(__name__)
        logging.debug('checking to start engine in ej')
        if not hasattr(self,'matlab_engine'):
            eng = matlab.engine.start_matlab()
            engine_name = eng.engineName
            logging.debug('new engine name:'+str(engine_name))
            a=eng.factorial(8)
            logging.debug('test using engine:8! ='+str(a))
            self.matlab_engine = eng
            logging.debug('ej engine:'+str(self.matlab_engine))
        else:
            logger.info('found engine in ej')
            logging.debug('found engine in ej')

        return self.perform_job(*args, **kwargs)

    def perform_job(self, job):
        """Performs the actual work of a job.  Will/should only be called
        inside the work horse's process.
        """
        self.prepare_job_execution(job)

        with self.connection._pipeline() as pipeline:
            started_job_registry = StartedJobRegistry(job.origin, self.connection)

            try:
                logging.debug('perform_job in sw')
                job.matlab_engine = self.matlab_engine
                logging.debug('pj engine:'+str(self.matlab_engine))
             #   logging.debug('pj args,kwargs:'+str(job._args)+','+str(job._kwargs))
                if len(job._args) > 0:
                    new_args = (self.matlab_engine,)+job._args
                    logging.debug('tg pj  new args:'+str(new_args))
                    job._args = new_args
                elif len(job._kwargs) > 0:
                    job._kwargs['matlab_engine']=self.matlab_engine
                    logging.debug('tg pj new kwargs:'+str(job._kwargs))
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
