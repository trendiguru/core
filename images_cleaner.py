__author__ = 'yonatan'

import sys

from termcolor import colored

from . import constants

db = constants.db
blackList = constants.blacklisted_terms
exceptions = constants.blacklisted_exceptions


def cleanMe(manual=False):
    images = db.images.find().batch_size(10000)
    print colored("total images before: %s" % db.images.count(), 'yellow')
    i = 0
    d = 0
    for doc in images:
        print colored('item #%s' % i, 'green')
        urls = doc['image_urls']
        for url in urls:
            if any(term in url for term in blackList):
                if any(x in url for x in exceptions):
                    continue
                if manual:
                    raw_input(url)
                else:
                    print(url)

                image_id = doc["_id"]
                db.images.delete_one({'_id': image_id})
                d += 1
                print colored("item #%s deleted!  delete count is %s" % (i, d), 'red', 'on_yellow')
                break
        i += 1

    print colored("total images after: %s" % db.images.count(), 'yellow')


if __name__ == "__main__":
    if len(sys.argv) == 2:
        manual = sys.argv[1]
    else:
        manual = False
    cleanMe(manual)
