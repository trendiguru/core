import gevent
from gevent.queue import Queue
tasks = Queue(maxsize=100)

