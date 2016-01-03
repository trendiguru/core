__author__ = 'yonatan'

"""
updates the status key of all the products in a collection
logic:
1. checks the date when it was last updated (the 'download_data.dl_version' value)
2. if it was updated in the last 2 days then it is instock -> "status.instock = True"
3. else "status.instock = False"
    and the days diff is calculated

* the flipkart collection is downloaded every 8 hours - we might want to shorten the 2 days margin to X hours.

"""
import datetime
import sys

from termcolor import colored

from . import constants

db = constants.db
month = 30


def update_status(coll="products"):
    collection = db[coll]
    today = datetime.datetime.today()
    today_date = str(today.date())
    yesterday = str((today - datetime.timedelta(1)).date())
    collection.update_many({"download_date": {"$not": today_date}}, {'$set': {"status.instock": False}})
    collection.update_many({"download_date": yesterday}, {'$set': {"status.instock": True}})
    total = collection.count()
    print colored("total = %s" % str(total), "red")
    instock = collection.find({"status.instock": True}).count()
    print colored("instock = %s" % str(instock), "blue")
    out = collection.find({"status.instock": False}).count()
    print colored("out of stock = %s" % str(out), "blue")
    sanity = total - instock
    if sanity == out:
        print colored("senity check ok", "green")
    else:
        print colored("this is insane", "red", "on_yellow")
    for day in range(2, month):
        date = str((today - datetime.timedelta(day)).date())
        res = collection.update_many({"download_date": date}, {'$set': {"status.days_out": day}})
        print colored("%s items out of stock for %s days " % (str(res.matched_count), str(day)), "magenta")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        update_status(sys.argv[1])
    else:
        update_status()
