__author__ = 'Nadav Paz'

import logging
import multiprocessing as mp
import argparse
import pymongo
import numpy as np
import cv2
import fingerprint_core as fp
import background_removal
import Utils
import constants
import time
import signal
import traceback


# globals
CLASSIFIER_FOR_CATEGORY = {}
TOTAL_PRODUCTS = mp.Value("i", 0)
CURRENT = Utils.ThreadSafeCounter()
DB = None
FP_VERSION = 0
START_TIME = 0
CONTINUE = mp.Value("b", True)
Q = mp.Queue(1000)
MAIN_PID = 0
NUM_PROCESSES = mp.Value("i", 0)


def get_all_subcategories(category_collection, category_id):
    """
    create a list of all subcategories in category_id, including itself.
    assumes category_collection is a mongodb Collection of category dictionaries
    with keys "id" and "childrenIds"
    :param category_collection: mongodb Collection
    :param category_id: string
    :return: list of all subcategories in category_id, including itself.
    """
    subcategories = []

    def get_subcategories(c_id):
        subcategories.append(c_id)
        curr_cat = category_collection.find_one({"id": c_id})
        if "childrenIds" in curr_cat.keys():
            for childId in curr_cat["childrenIds"]:
                get_subcategories(childId)

    get_subcategories(category_id)
    return subcategories


def create_classifier_for_category_dict(db):
    """
    Creates a dictionary with items: category_id: CascasdeClassifier
    Requires cv2 and constants to be imported
    :param db: connected pymongo.MongoClient().db object
    :return: dictionary with items: category_id: CascasdeClassifier
    """
    result_dict = {}
    classifier_dict = {xml: cv2.CascadeClassifier(constants.classifiers_folder + xml)
                       for xml in constants.classifier_to_category_dict.keys()}
    for xml, cats in constants.classifier_to_category_dict.iteritems():
        for cat in cats:
            for sub_cat in get_all_subcategories(db.categories, cat):
                result_dict[sub_cat] = classifier_dict[xml]
    return result_dict


def run_fp(doc):
    CURRENT.increment()
    if CURRENT.value % 100 == 0:
        print "Process {process} starting {i} of {total}...".format(process=mp.current_process().name,
                                                                    i=CURRENT.value, total=TOTAL_PRODUCTS.value)
    image_url = doc["image"]["sizes"]["XLarge"]["url"]
    image = Utils.get_cv2_img_array(image_url)
    if not Utils.is_valid_image(image):
        logging.warning("image is None. url: {url}".format(url=image_url))
        return
    small_image, resize_ratio = background_removal.standard_resize(image, 400)
    # I think we can delete this... memory management FTW??
    del image
    # print "Image URL: {0}".format(image_url)
    # if there is a valid human BB, use it
    if "human_bb" in doc.keys() and doc["human_bb"] != [0, 0, 0, 0] and doc["human_bb"] is not None:
        chosen_bounding_box = doc["human_bb"]
        chosen_bounding_box = [int(b) for b in (np.array(chosen_bounding_box) / resize_ratio)]
        mask = background_removal.get_fg_mask(small_image, chosen_bounding_box)
        logging.debug("Human bb found: {bb} for item: {id}".format(bb=chosen_bounding_box, id=doc["id"]))
    # otherwise use the largest of possibly many classifier bb's
    else:
        if "categories" in doc:
            classifier = CLASSIFIER_FOR_CATEGORY.get(doc["categories"][0]["id"], "")
        else:
            classifier = None

        # first try grabcut with no bb
        if not Utils.is_valid_image(small_image):
            logging.warning("small_image is Bad. {img}".format(img=small_image))
            return
        mask = background_removal.get_fg_mask(small_image)
        bounding_box_list = []

        if classifier and not classifier.empty():
            # then - try to classify the image (white backgrounded and get a more accurate bb
            white_bckgnd_image = background_removal.image_white_bckgnd(small_image, mask)
            try:
                bounding_box_list = classifier.detectMultiScale(white_bckgnd_image)
            except KeyError:
                logging.info("Could not classify with {0}".format(classifier))
        # choosing the biggest bounding box if there are a few
        max_bb_area = 0
        chosen_bounding_box = None
        for possible_bb in bounding_box_list:
            if possible_bb[2] * possible_bb[3] > max_bb_area:
                chosen_bounding_box = possible_bb
                max_bb_area = possible_bb[2] * possible_bb[3]
        if chosen_bounding_box is None:
            logging.info("No Bounding Box found, using the whole image. "
                         "Document id: {0}, BB_list: {1}".format(doc.get("id"), str(bounding_box_list)))
        else:
            mask = background_removal.get_fg_mask(small_image, chosen_bounding_box)
    try:
        fingerprint = fp.fp(small_image, mask)
        DB.products.update({"id": doc["id"]},
                           {"$set": {"fingerprint": fingerprint.tolist(),
                                     "fp_version": FP_VERSION,
                                     "bounding_box": np.array(chosen_bounding_box).tolist()}})

    except Exception as ex:
        logging.warning("Exception caught while fingerprinting: {0}".format(ex))


def do_work_on_q(some_func, q):
    current_pid = mp.current_process().pid
    print "{0} Getting ready to do some work...".format(str(current_pid))
    try:
        while CONTINUE.value:
            popped_item = q.get()
            if popped_item is None:
                print "Process {0} finished".format(str(current_pid))
                return

            some_func(popped_item)
    except BaseException as be:
        print "Process {0}, exception reached do_work:\n".format(str(current_pid))
        traceback.print_exc()
        # get back to work
        do_work_on_q(some_func, q)
    print "{0} all done...".format(str(current_pid))
    return "{0} returned".format(str(current_pid))


def connect_db_feed_q(q, query_doc, fields_doc):
    """
    Connects to the DB, queries, and fills q with results.
    Also sets global TOTAL_PRODUCTS, DB
    :param q:
    :return:
    """
    global TOTAL_PRODUCTS, DB
    DB = DB or pymongo.MongoClient().mydb
    product_cursor = DB.products.find(query_doc, fields_doc)  # .batch_size(n)

    TOTAL_PRODUCTS.value = product_cursor.count()
    print "Total tasks: {0}".format(str(TOTAL_PRODUCTS.value))

    for doc in product_cursor:
        q.put(doc)

    for p in range(0, NUM_PROCESSES.value):
        q.put(None)

    print "Done putting all docs in Q"
    q.close()


def print_stats(start_time):
    stop_time = time.time()
    total_time = stop_time - start_time
    print "Stats:\n " \
          "Completed {total} fingerprints in {seconds} seconds with {procs} processes.\n " \
          "Average time per fingerprint: {avg}\n " \
          "Average time per fingerprint per core: {avgc}\n"\
        .format(avg=total_time/CURRENT.value, total=CURRENT.value,
                seconds=total_time, procs=NUM_PROCESSES.value,
                avgc=(total_time/CURRENT.value)*NUM_PROCESSES.value)


def fingerprint_db(fp_version, category_id=None, num_processes=None):
    """
    main function - fingerprints items in category_id and its subcategories.
     If category_id is None, then fingerprints entire db. Also manages the multiprocessing
    :param fp_version: integer to keep track of which items have been already fingerprinted with this version
    :param category_id: category to be fingerprinted
    :return:
    """
    global CURRENT, CLASSIFIER_FOR_CATEGORY, FP_VERSION, NUM_PROCESSES, DB, START_TIME, MAIN_PID

    MAIN_PID = mp.current_process().pid

    NUM_PROCESSES.value = num_processes or int(mp.cpu_count() * 0.75)

    DB = DB or pymongo.MongoClient().mydb
    if category_id is not None:
        query_doc = {"$and": [
            {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(DB.categories, category_id)}}}},
            {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}
        ]}
    else:
        query_doc = {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}

    fields = {"image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1, "categories": 1, "id": 1}

    CLASSIFIER_FOR_CATEGORY = create_classifier_for_category_dict(DB)
    FP_VERSION = fp_version

    feeder = mp.Process(target=connect_db_feed_q, name="Feeder", args=[Q, query_doc, fields])
    worker_list = [mp.Process(target=do_work_on_q, name="Worker {0}".format(p), args=(run_fp, Q))
                   for p in range(0, NUM_PROCESSES.value)]

    START_TIME = time.time()
    feeder.start()

    for p in worker_list:
        p.start()

    for p in worker_list:
        p.join()

    feeder.join()

    stop_time = time.time()
    total_time = stop_time - START_TIME

    print "All done!!"
    print "Completed {total} fingerprints in {seconds} seconds " \
          "with {procs} processes.".format(total=TOTAL_PRODUCTS.value, seconds=total_time, procs=num_processes)
    print "Average time per fingerprint: {avg}".format(avg=total_time/TOTAL_PRODUCTS.value)
    print "Average time per fingerprint per core: {avgc}".format(avgc=(total_time/TOTAL_PRODUCTS.value)*num_processes)


def fingerprint_db_old(fp_version, category_id=None, num_processes=None):
    """
    main function - fingerprints items in category_id and its subcategories.
     If category_id is None, then fingerprints entire db.
    :param fp_version: integer to keep track of which items have been already fingerprinted with this version
    :param category_id: category to be fingerprinted
    :return:
    """
    global DB, TOTAL_PRODUCTS, CURRENT, CLASSIFIER_FOR_CATEGORY, FP_VERSION

    DB = DB or pymongo.MongoClient().mydb
    num_processes = num_processes or mp.cpu_count() - 2

    if category_id is not None:
        query_doc = {"$and": [
            {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(DB.categories, category_id)}}}},
            {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}
        ]}
    else:
        query_doc = {"$or": [{"fp_version": {"$lt": fp_version}}, {"fp_version": {"$exists": 0}}]}

    fields = {"image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1, "categories": 1, "id": 1}

    # batch_size required because cursor timed out without it. Could use further investigation
    product_cursor = DB.products.find(query_doc, fields).batch_size(num_processes)
    TOTAL_PRODUCTS = product_cursor.count()
    CLASSIFIER_FOR_CATEGORY = create_classifier_for_category_dict(DB)

    FP_VERSION = fp_version

    pool = mp.Pool(num_processes, maxtasksperchild=5)

    start_time = time.time()
    pool.map(run_fp, product_cursor)
    stop_time = time.time()
    total_time = stop_time - start_time
    pool.close()
    pool.join()

    print "All done!!"
    print "Completed {total} fingerprints in {seconds} seconds " \
          "with {procs} processes.".format(total=TOTAL_PRODUCTS, seconds=total_time, procs=num_processes)
    print "Average time per fingerprint: {avg}".format(avg=total_time/TOTAL_PRODUCTS)
    print "Average time per fingerprint per core: {avgc}".format(avgc=(total_time/TOTAL_PRODUCTS)*num_processes)


def receive_signal(signum, stack):
    if signum == 17 or 28:
        # 17 creating child process, ignore
        # 28 SIGWINCH, ignore
        return
    if signum == 2 and mp.current_process().pid == MAIN_PID:
        print_stats(START_TIME)
        return
    print '{0} caught signal {1}.'.format(mp.current_process().pid, str(signum))
    traceback.print_stack(stack)


if __name__ == "__main__":

    uncatchable = ['SIG_DFL', 'SIGSTOP', 'SIGKILL']
    for i in [x for x in dir(signal) if x.startswith("SIG")]:
        if not i in uncatchable:
            signum = getattr(signal, i)
            signal.signal(signum, receive_signal)

    try:
        parser = argparse.ArgumentParser(description='Fingerprint the DB or part of it')
        parser.add_argument('-c', '--category_id', help='id of category to be fingerprinted', required=False)
        parser.add_argument('-p', '--num_processes', help='number of parallel processes to spawn',
                            required=False, type=int)
        parser.add_argument('-v', '--fp_version', help='current fp version', required=True)
        args = vars(parser.parse_args())

        fingerprint_db(int(args['fp_version']), args['category_id'], args['num_processes'])
    except Exception as e:
        logging.warning("Exception reached main!: {0}".format(e))
