__author__ = 'yonatan'
import time
import random

from termcolor import colored
from selenium import webdriver
from selenium.webdriver.common.proxy import *
from rq import Queue

import constants

redis_conn = constants.redis_conn_old
new_images_Q = Queue("new_images", connection=redis_conn)
paperdoll_Q = Queue("pd", connection=redis_conn)


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
    # subprocess.call(["sudo", "rm", "-r", "/tmp/tmp*"], shell=True)
    # from xvfbwrapper import Xvfb
    # from pyvirtualdisplay import Display
    print colored("Running Extension on %s" % url, "blue", "on_white")

    #
    # # enable browser logging
    # display = Display(visible=0, size=(1024, 786))
    # display.start()
    # newProxy = getProxy()
    # driver = webdriver.Firefox(proxy=newProxy)
    # xvfb.start()
    driver = webdriver.Firefox(
        firefox_binary=webdriver.firefox.firefox_binary.FirefoxBinary(
            log_file=open('/home/yonatan/selenium.log', 'a')))
    try:

        driver.get(url)
        scr = open("/var/www/latest/b_main.js").read()

        # wait for the queues to be empty enough
        while paperdoll_Q.count > 50 & new_images_Q.count > 50000:
            time.sleep(5)

        driver.execute_script(scr)
        time.sleep(1)
        for x in range(8):
            script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
            driver.execute_script(script)
            time.sleep(0.25)
        print colored("execute Success!!!", "green", "on_white")
    except:
        print colored("execute Failed!!!", "red", "on_yellow")
    try:
        driver.quit()
    except:
        print colored("driver.quit() Failed!!!", "red", "on_yellow")
        # display.popen.terminate()
        # xvfb.stop()
        # display.stop()


proxies = [['118.142.33.112', '8088'],
           ['31.173.74.73', '8080'],
           ['198.169.246.30', '80'],
           ['202.29.97.2', '3128'],
           ['91.121.181.168', '80']]
