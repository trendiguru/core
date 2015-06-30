__author__ = 'jeremy'

import requests

from redis import Redis
from rq import Queue


def count_words_at_url(url):
    resp = requests.get(url)
    return len(resp.text.split())


def wrap():
    url = 'http://www.cnn.com'
    q = Queue(connection=Redis())
    result = q.enqueue(count_words_at_url, 'http://nvie.com')

    print(str(result) + ' words at ' + url)

if __name__ == '__main__':
    print('starting')
    # kill_images_collection()
    # verify_hash_of_image('wefwfwefwe', 'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d.jpg')
    wrap()