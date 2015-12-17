__author__ = 'yonatan'
import time

from selenium import webdriver
from pyvirtualdisplay import Display

def runExt(url):
    # enable browser logging
    display = Display(visible=0, size=(1024, 768))
    display.start()
    driver = webdriver.Firefox()
    driver.get(url)
    scr = open("/var/www/latest/b_main.js").read()
    driver.execute_script(scr)
    time.sleep(5)
    for x in range(10):
        script = "scroll(" + str(x * 500) + "," + str(x * 500 + 500) + ")"
        driver.execute_script(script)
        time.sleep(0.5)

    driver.close()
    display.stop()
