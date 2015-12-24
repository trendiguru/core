__author__ = 'yonatan'
import time
import random

from termcolor import colored
from selenium import webdriver
from selenium.webdriver.common.proxy import *


def getProxy():
    pro = random.sample(proxies, 1)
    myProxy = pro[0][0] + ":" + pro[0][1]

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
    from pyvirtualdisplay import Display
    print colored("Running Extension on %s" % url, "yellow")

    # enable browser logging
    display = Display(visible=0, size=(1024, 768))
    display.start()
    newProxy = getProxy()
    driver = webdriver.Firefox(proxy=newProxy)
    try:
        driver.get(url)
        scr = open("/var/www/latest/b_main.js").read()

        driver.execute_script(scr)
        time.sleep(1)
        for x in range(8):
            script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
            driver.execute_script(script)
            time.sleep(0.25)
        print colored("execute Success!!!", "green")
    except:
        print colored("execute Failed!!!", "red")

    driver.quit()
    display.popen.terminate()


proxies = [['118.142.33.112', '8088'],
           ['31.173.74.73', '8080'],
           ['198.169.246.30', '80'],
           ['202.29.97.2', '3128'],
           ['91.121.181.168', '80']]
