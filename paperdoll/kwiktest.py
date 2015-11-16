__author__ = 'jeremy'
from trendi.paperdoll import paperdoll_parse_enqueue
import time
import numpy as np
urls=[]
dt=[]
urls.append('http://i.imgur.com/ahFOgkm.jpg')
urls.append('https://img1.etsystatic.com/019/1/5682424/il_570xN.555916317_ebv0.jpg')
for url in urls:
    start_time = time.time()
    retval = paperdoll_parse_enqueue.paperdoll_enqueue(url, async=False)
    end_time = time.time()
    dt=end_time-start_time
    dts.append(dt)
    print('retval:' + str(retval.result)+' time:'+str(dt))
means=np.mean(dt)
std=np.std(dt)
print('mean:' + str(means)+' std:'+str(std))
