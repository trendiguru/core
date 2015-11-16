__author__ = 'jeremy'
from trendi.paperdoll import paperdoll_parse_enqueue
import time
import numpy as np
urls=[]
dts=[]

urls.append('http://i.imgur.com/ahFOgkm.jpg')
urls.append('https://img1.etsystatic.com/019/1/5682424/il_570xN.555916317_ebv0.jpg')
urls.append('http://www.customcelebritydresses.com/image/cache/data/4.13/Christina%20Hendricks%20Green%20V-neck%20Prom%20Dress%202008%20Emmy%20Awards%20Red%20Carpet%202-600x600.jpg')
urls.append('http://www.hollywoodtuna.com/images/christina_hendricks_green_small.jpg')
urls.append('http://aws-cdn.dappered.com/wp-content/uploads/2014/03/CH2008.jpg')
urls.append('http://www.fashiontrendspk.com/wp-content/uploads/Emma-Stone-in-Lanvin-at-the-2012-Golden-Globe-Awards-450x345.jpg')
urls.append('http://gingertalk.com/wp-content/uploads/2013/12/golddress-200x300.jpg')
urls.append('https://s-media-cache-ak0.pinimg.com/736x/fb/92/50/fb9250e68e63f6862da24bfb3ae17b0a.jpg')
urls.append('https://s-media-cache-ak0.pinimg.com/736x/71/82/42/7182428f1c4b584fe084823791aa9d59.jpg')
urls.append('https://s-media-cache-ak0.pinimg.com/736x/c1/a4/56/c1a456661a699c99e9a019648ba928a2.jpg')
urls.append('http://gingerparrot.co.uk/wp/wp-content/uploads/2014/04/Katy-B-Red-Hair-White-Clothes-Still-Video.jpg')
urls.append('http://media2.popsugar-assets.com/files/2010/08/34/5/192/1922153/9621f2d8749ddac7_red-main/i/What-Kind-Makeup-Wear-Youre-Redhead-Wearing-Red-Dress.jpg')


for url in urls:
    start_time = time.time()
    retval = paperdoll_parse_enqueue.paperdoll_enqueue(url, async=False,use_parfor=True,queue_name='pd_parfor')
    end_time = time.time()
    dt=end_time-start_time
    dts.append(dt)
    print('retval:' + str(retval.result)+' time:'+str(dt))
means=np.mean(dts)
std=np.std(dts)
print('mean:' + str(means)+' std:'+str(std))
