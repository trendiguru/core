import os

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = os.environ["REDIS_PORT"]

REDIS_URL = 'redis://{0}:{1}/1'.format(REDIS_HOST, REDIS_PORT)

