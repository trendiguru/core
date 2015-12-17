__author__ = 'yonatan'
import time

from selenium import webdriver


def runExt(url):
    # enable browser logging
    driver = webdriver.Chrome(executable_path="/usr/src/linux-headers-3.16.0-55/drivers/platform/chrome",
                              service_log_path="/home/developer/ghostdriver.log")
    driver.get(url)
    scr = open("/var/www/latest/b_main.js").read()
    driver.execute_script(scr)
    time.sleep(5)
    for x in range(10):
        script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
        driver.execute_script(script)
        time.sleep(0.5)

    driver.quit()
