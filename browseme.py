__author__ = 'yonatan'
import time

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


def runExt(url):
    # enable browser logging

    user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:16.0) Gecko/20121026 Firefox/16.0"

    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] = user_agent

    driver = webdriver.PhantomJS(desired_capabilities=dcap, service_log_path="/home/developer/ghostdriver.log")
    driver.get(url)
    scr = open("/var/www/latest/b_main.js").read()
    driver.execute_script(scr)
    time.sleep(1)
    for x in range(10):
        script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
        driver.execute_script(script)
        time.sleep(0.5)

    driver.quit()
