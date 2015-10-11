__author__ = 'jeremy'



import unittest
import time
from rq import Queue
from redis import Redis

from trendi_guru_modules.paperdoll import paperdoll_parse_enqueue

redis_conn = Redis()


class OutcomesTest(unittest.TestCase):
    # examples of things to return
    #    def testPass(self):
    #        return
    #    def testFail(self):
    #        self.failIf(True)
    #    def testError(self):
    #        raise RuntimeError('Test error!')
    #run a timing test
    def test_bad_url(self):
        url = 'http://notanimage.jpg'
        queue = Queue('paperdoll', connection=redis_conn)

        print('testing bad url:'+url)
    #    img, labels, pose = paperdoll_enqueue(url, async = True,queue=queue)
         #n = paperdoll_enqueue(url, async = True,queue=queue)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False)
        print('labels:'+str(labels))
        print('')

        print('testing bad url:'+url)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = True)
        print('labels:'+str(labels))
        print('')

    def test_tg_and_regular_worker(self):
        url = 'http://notanimage.jpg'
        queue = Queue('paperdoll', connection=redis_conn)

        print('testing tg worker on:'+url)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False,use_tg_worker=True)
        print('labels:'+str(labels))
        print('')

        print('testing regular redis worker on:'+url)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False,use_tg_worker=False)
        print('labels:'+str(labels))
        print('')

        url = 'http://i.imgur.com/ahFOgkm.jpg'
        print('testing tg worker on:'+url)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False,use_tg_worker=True)
        print('labels:'+str(labels))
        print('')

        print('testing regular redis worker on:'+url)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False,use_tg_worker=False)
        print('labels:'+str(labels))
        print('')

    #run a timing test
    def test_time(self):
        urls = ['http://i.imgur.com/ahFOgkm.jpg',\
                'http://www.wantdresses.com/wp-content/uploads/2015/09/sexy-plus-sized-prom-dresses-at-peaches-boutique-peaches-awesome-prom-dresses.jpg',\
                'http://www.wantdresses.com/wp-content/uploads/2015/07/skvdfw-l.jpg',\
                'http://www.wantdresses.com/wp-content/uploads/2015/09/xoxo-prom-dresses-awesome-prom-dresses.jpg',\
                'http://www.wantdresses.com/wp-content/uploads/2015/09/prom-dresses-on-pinterest-orange-prom-dresses-pink-prom-dresses-awesome-prom-dresses.jpg',\
                'http://www.wantdresses.com/wp-content/uploads/2015/09/fun-prom-dresses-2013-look-awesome-in-ombre-blog-at-the-prom-awesome-prom-dresses.jpg',\
                'http://www.wantdresses.com/wp-content/uploads/2015/09/dress-jewellery-picture-more-detailed-picture-about-hot-sale-awesome-prom-dresses.jpg',\
                'http://www.wantdresses.com/wp-content/uploads/2015/07/rs_634x926-140402114112-634-8Prom-Dress-ls.4214.jpg',\
                'http://www.wantdresses.com/wp-content/uploads/2015/09/group-of-vsledky-obrzk-google-pro-httpwwwoblectesecz-awesome-prom-dresses.jpg',\
                'http://www.wantdresses.com/wp-content/uploads/2015/09/gowns-blue-picture-more-detailed-picture-about-awesome-strapless-awesome-prom-dresses.jpg']
        i = 0
        start_time = time.time()
        queue = Queue('paperdoll_test', connection=redis_conn)

        for url in urls:
            i+=1
            print('url #'+str(i)+' '+url)
            img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False,use_tg_worker=True)
    #        n = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = True,queue=queue)
           # img, labels, pose = paperdoll_enqueue_parallel(url, async = True)
            print('labels:'+str(labels))
            print('')
        elapsed_time = time.time() - start_time
        print('tot elapsed:'+str(elapsed_time)+',per image:'+str(float(elapsed_time)/len(urls)))


if __name__ == '__main__':
    unittest.main()
