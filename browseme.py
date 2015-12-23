__author__ = 'yonatan'
import time

from termcolor import colored
from selenium import webdriver


def runExt(url):
    from pyvirtualdisplay import Display
    print colored("Running Extension on %s" % url, "yellow")

    # enable browser logging
    display = Display(visible=0, size=(1024, 768))
    display.start()
    driver = webdriver.Firefox()
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
