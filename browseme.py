__author__ = 'yonatan'
import time

from selenium import webdriver


def runExt(url):
    # enable browser logging
    driver = webdriver.PhantomJS(service_args=["--webdriver-loglevel=ERROR"])
    driver.set_window_size(1120, 1050)
    driver.get(url)
    driver.execute_script(open("/var/www/latest/b_main.js").read())
    time.sleep(1)
    for x in range(10):
        script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
        driver.execute_script(script)
        time.sleep(0.5)

    driver.quit()
