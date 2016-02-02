__author__ = 'yonatan'
import time
import random
import os

from termcolor import colored
from selenium import webdriver
from selenium.webdriver.common.proxy import *
from rq import Queue
from redis import Redis










# import constants

redis_conn = Redis(host="redis1-redis-1-vm")  # constants.redis_conn
pipeline_Q = Queue("start_pipeline", connection=redis_conn)
paperdoll_Q = Queue("pd", connection=redis_conn)
browse_q = Queue('BrowseMe', connection=redis_conn)


def getProxy():
    pro = random.sample(proxies, 1)
    myProxy = pro[0][0] + ":" + pro[0][1]
    print colored('using Proxy = ' + myProxy, 'magenta', attrs=['bold'])
    proxy = Proxy({
        'proxyType': ProxyType.MANUAL,
        'httpProxy': myProxy,
        'ftpProxy': myProxy,
        'sslProxy': myProxy,
        'noProxy': None,
        'autodetect': False
    })

    return proxy


def runExt(url):
    # if get_count() > 10:
    #     try:
    #         ret = subprocess.call(["sudo rm -r /tmp/.X*"], shell=True)
    #         if not ret:
    #             print colored("remove X success", "yellow")
    #         else:
    #             print colored("remove X failed", "magenta")
    #     except:
    #         pass


    # from xvfbwrapper import Xvfb
    # from pyvirtualdisplay import Display
    print colored("Running Extension on %s" % url, "magenta", attrs=['bold'])

    #
    # # enable browser logging
    # display = Display(visible=0, size=(1024, 786))
    # display.start()
    # newProxy = getProxy()

    # xvfb.start()
    # driver = webdriver.Firefox(
    #     firefox_binary=webdriver.Firefox.firefox_binary.FirefoxBinary(
    #         log_file=open('/home/yonatan/selenium.log', 'a')))
    # tmpGuard.main()
    driver = webdriver.Firefox()
    try:
        scr = open("/var/www/latest/b_main.js").read()
        print colored("driver started", "yellow")
        # wait for the queues to be empty enough
        countQue = 0
        while paperdoll_Q.count > 250 or pipeline_Q.count > 5000:
            countQue += 1
            if countQue > 2:
                print colored("Que Full - returned to Que", "green", attrs=['bold'])
                browse_q.enqueue(runExt, url)
                # driver.quit()
                return
            print colored("Que Full - taking 15 sec break", "red")
            time.sleep(15)
        print colored("0", "green")
        driver.get(url)
        print colored("1", "green")
        driver.execute_script(scr)
        print colored("2", "green")
        time.sleep(1)
        print colored("script executed!", "green")

        for x in range(8):
            script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
            driver.execute_script(script)
            time.sleep(0.25)
        print colored("execute Success!!!", "yellow", "on_magenta", attrs=['bold'])
    except:
        print colored("execute Failed!!!", "red", "on_yellow", attrs=['bold'])
    try:
        driver.quit()
    except:
        print colored("driver.quit() Failed!!!", "red", "on_yellow", attrs=['bold'])
        # display.popen.terminate()
        # xvfb.stop()
        # display.stop()
        # try:
        #     ret = subprocess.call(["sudo rm -r /tmp/tmp*"], shell=True)
        #     if not ret:
        #         print colored("remove tmp success", "yellow")
        #     else:
        #         print colored("remove tmp failed", "magenta")
        # except:
        #     pass

proxies = [['118.142.33.112', '8088'],
           ['31.173.74.73', '8080'],
           ['198.169.246.30', '80'],
           ['202.29.97.2', '3128'],
           ['91.121.181.168', '80']]


def get_count(start_path='.'):
    repeat = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # total_size += os.path.getsize(fp)
            print f
            if f[:3] == 'tmp':
                repeat += 1
    return repeat
