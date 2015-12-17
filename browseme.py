__author__ = 'yonatan'
import time

from selenium import webdriver
from pyvirtualdisplay import Display


def runExt(url):
    print("Running Extension on %s" % url)
    # enable browser logging
    display = Display(visible=0, size=(1024, 768))
    display.start()
    driver = webdriver.Firefox()
    driver.get(url)
    scr = open("/var/www/latest/b_main.js").read()
    driver.execute_script(scr)
    time.sleep(1)
    for x in range(8):
        script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
        driver.execute_script(script)
        time.sleep(0.25)

    driver.quit()
    display.popen.terminate()
