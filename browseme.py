__author__ = 'yonatan'
import time

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


def runExt(url):
    # enable browser logging

    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.71 Safari/537.36'

    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] = user_agent

    driver = webdriver.PhantomJS(desired_capabilities=dcap, service_log_path="/home/developer/ghostdriver.log")
    driver.set_window_size(1024, 768)

    driver.get(url)
    scr = open("/var/www/latest/b_main.js").read()
    print scr
    driver.execute_script(scr)
    time.sleep(1)
    for x in range(10):
        script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
        driver.execute_script(script)
        time.sleep(0.5)

    driver.quit()
