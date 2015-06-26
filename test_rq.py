__author__ = 'liorsabag'

import requests
import time

def count_words_at_url(url):
    resp = requests.get(url)
    return len(resp.text.split())

def delayed_print(obj):
    print "Starting"
    time.sleep(5)
    print obj
    return True