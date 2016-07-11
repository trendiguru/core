from .. import constants
import pymongo
import subprocess

db = constants.db
prefix = 'https://storage.googleapis.com/'


def overflow_test():
    irrelevant_urls = []
    # create list of ir/relevant images urls from db.irrelevant_/images &
    curs = db.images.find({}, {'image_urls': 1}).sort('_id', pymongo.ASCENDING).limit(1000)
    relevant_urls = get_first_x_images_from_collection(2000, 'images') + \
                    get_urls_from_gs("gs://tg-training/doorman/relevant")
    print "done collecting relevant-images, total {0}".format(len(relevant_urls))
    irrelevant_urls = get_urls_from_gs("gs://tg-training/doorman/irrelevant")

    p = subprocess.Popen(["gsutil", "ls", "gs://tg-training/doorman/irrelevant/irrelevant_images_for_doorman"], stdout=subprocess.PIPE)
    # divide to batches of random relevant/irrelevant images
    # simulate reasonable POST requests tempo to https://api.trendi.guru/images


def get_first_x_images_from_collection(x, collection):
    curs = db[collection].find({}, {'image_urls': 1}).sort('_id', pymongo.ASCENDING).limit(x)
    return [doc['image_urls'][0] for doc in curs]


def get_urls_from_gs(storage_lib):
    p = subprocess.Popen(["gsutil", "ls", storage_lib], stdout=subprocess.PIPE)
    output, err = p.communicate()
    output = [prefix+url[17:] for url in output.split('\n')]
    return output
