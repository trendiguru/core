__author__ = 'jeremy'

__author__ = 'jeremy'



import unittest
import time
from rq import Queue

import paperdoll_parse_enqueue
import paperdolls
import redis_conn



class OutcomesTest(unittest.TestCase):
    # examples of things to return
    #    def testPass(self):
    #        return
    #    def testFail(self):
    #        self.failIf(True)
    #    def testError(self):
    #        raise RuntimeError('Test error!')
    #run a timing test

    def test_callback(self):
        url = 'http://i.imgur.com/ahFOgkm.jpg'
        print('testing callback async')
        retval = paperdoll_parse_enqueue.paperdoll_enqueue(url,async=True,use_tg_worker=False,callback_function=paperdolls.callback_example,args=(100,101),kwargs={'a':3,'b':4})
        print('retval:'+str(retval))

        print('testing callback sync')
        img,labels,pose = paperdoll_parse_enqueue.paperdoll_enqueue(url,async=False,use_tg_worker=False,callback_function=paperdolls.callback_example,args=(100,101),kwargs={'a':3,'b':4})
        print('labels:'+str(labels))


    def test_bad_url(self):
        url = 'http://notanimage.jpg'
        queue = Queue('paperdoll', connection=redis_conn)

        print('testing bad url (async False, tg_worker true):'+url)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False,use_tg_worker=True)
        print('labels:'+str(labels))
        print('')

        print('testing bad url:(async true, tg_worker true)'+url)
        retval = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = True,use_tg_worker=True)
        print('retval:'+str(retval))
        print('')

        print('testing bad url (async False, tg_worker False):'+url)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False,use_tg_worker=False)
        print('labels:'+str(labels))
        print('')

        print('testing bad url:(async true, tg_worker False)'+url)
        retval = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = True,use_tg_worker=False)
        print('retval:'+str(retval))
        print('')

    def test_tg_and_regular_worker(self):

        url = 'http://i.imgur.com/ahFOgkm.jpg'
        print('testing tg worker on (async False, tg_worker True):'+url)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False,use_tg_worker=True)
        print('labels:'+str(labels))
        print('')

        print('testing regular redis worker (async False, tg_worker False):'+url)
        img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = False,use_tg_worker=False)
        print('labels:'+str(labels))
        print('')

        print('testing tg worker (async True, tg_worker True)'+url)
        retval=paperdoll_parse_enqueue.paperdoll_enqueue(url, async = True,use_tg_worker=True)
        print('retval:'+str(retval))
        print('')

        print('testing regular redis worker  (async True, tg_worker False):'+url)
        retval = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = True,use_tg_worker=False)
        print('retval:'+str(retval))
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
        queue = Queue('paperdoll_test', connection=redis_conn)

        #test async
        for url in urls:
            i+=1
            print('url #'+str(i)+' '+url)
            retval = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = True,use_tg_worker=True)
            print('retval:'+str(retval))
    #        n = paperdoll_parse_enqueue.paperdoll_enqueue(url, async = True,queue=queue)
           # img, labels, pose = paperdoll_enqueue_parallel(url, async = True)
            print('')

        #test sync
        start_time = time.time()
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

