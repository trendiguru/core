__author__ = 'jeremy'
from trendi_guru_modules.paperdoll import paperdoll_parse_enqueue
url = 'http://i.imgur.com/ahFOgkm.jpg'
retval = paperdoll_parse_enqueue.paperdoll_enqueue(url,async=False)
print('retval:'+str(retval.result))