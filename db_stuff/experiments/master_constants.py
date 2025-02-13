import os
import pymongo
from redis import StrictRedis


db = pymongo.MongoClient(host=os.getenv("MONGO_HOST", "mongodb1-instance-1"),
                         port=int(os.getenv("MONGO_PORT", "27017"))).mydb

redis_url = os.environ.get("REDIS_URL", None)
if redis_url:
    redis_conn = StrictRedis.from_url(redis_url)
else:
    redis_conn = StrictRedis(host=os.getenv("REDIS_HOST", "redis1-redis-1-vm"),
                             port=int(os.getenv("REDIS_PORT", "6379")))

redis_limit = 5000

