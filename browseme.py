__author__ = 'yonatan'
import time

from selenium import webdriver


def runExt(url):
    # enable browser logging
    driver = webdriver.Firefox()  # (service_log_path="/home/developer/ghostdriver.log",service_args=["--webdriver-loglevel=ERROR"])
    driver.get(url)
    driver.execute_script(open("/var/www/latest/b_main.js").read())
    time.sleep(1)
    for x in range(10):
        script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
        driver.execute_script(script)
        time.sleep(0.5)

    driver.quit()
