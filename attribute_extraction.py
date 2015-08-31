__author__ = 'sergey'
# Examining the algorithm of "attribute_extraction idea":

import pickle
import collections
import os
import itertools

import html2text
import pymongo

from nlpcm13 import NLPCmatrix as nlpcm



# The first step is to build the input structures:
#  1. list of description_tuples (description id, description content).
#  2. From already built list of description construct the list of collocations.
# The second step is to save them into pickle files.

description_tuple = collections.namedtuple("description_tuple", ["id", "description"])


def html_to_str(h_str):
    """
    :param h_str: html code
    :return: a "clear" string without signs of html language.
    """
    str_m = html2text.html2text(h_str)
    str_m = str_m.encode('ascii', 'ignore')  # conversion of unicode type to string type
    return str_m


def get_all_subcategories(category_collection, category_id):
    """
    __author__ = TrendiGuru:
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


def get_all_dresses_data():
    """
    This function finds only data of dresses.
    :return: cursor to data of dresses (only id, categories and description)
    """
    db = pymongo.MongoClient().mydb
    category_id = "dresses"
    query = {"categories": {"$elemMatch": {"id": {"$in": \
                                                      get_all_subcategories(db.categories, category_id)}}}}
    fields = {"categories": 1, "id": 1, "description": 1}
    data = db.products.find(query, fields)
    return data


def get_descriptions(data):
    """
    :param data: cursor to a data set.
    :return: list of descriptions (list of description_tuple-s).
    """
    count = 0
    descs_list = []
    for product in data:
        d_tuple = description_tuple(id=product["id"], description=html_to_str(product["description"]))
        descs_list.append(d_tuple)
        count += 1
        print count
    return descs_list


def save(desc_list, file_dir="descriptions"):
    """
    :param desc_list: any variable which will be saved.
    :param file_dir: the directory (without file type) where
           the obtainable variable will be saved as pickle file.
    :return: None
    """
    with open(file_dir + ".p", "wb") as f:
        pickle.dump(desc_list, f)
    f.close()


def write_into_txt(desc_list, direct):
    with open(direct, "w") as f:
        for desc in desc_list:
            f.writelines(desc.description)
        f.close()


def attribute_extraction(keys_list, desc_list):
    """
    The main purpose of the function is to find collocations of words which are un-intersected
    (if in description presents one word the another is not here and conversely.)
    :param keys_list: list of keys (tuples(word, number)).
    :param desc_list: list of all descriptions from which the keys list was built.
    :return:
    """
    word_dict = {}
    for key_tuple in keys_list:
        desc_id_set = set()
        for desc in desc_list:
            if key_tuple[0] in desc.description:
                desc_id_set.add(desc.id)
        word_dict[key_tuple[0]] = desc_id_set
    print word_dict
    # for every pair of words in word_list,
    # find the size of the intersection of their description_set's
    intersection_dict = {}
    for word_a, word_b in itertools.combinations(word_dict.keys(), 2):
        intersection_dict[(word_a, word_b)] = len(word_dict[word_a].intersection(word_dict[word_b]))
    print intersection_dict
    return intersection_dict


#  block of main functions:
def main_function1():
    #   first step:
    d = get_descriptions(get_all_dresses_data())
    save(d, "list_of_dresses_descriptions.p")


def main_function2():
    # second step:
    f = open(os.path.dirname(os.path.realpath(__file__)) + "\\list_of_dresses_descriptions.p", "r")
    descr_list = pickle.load(f)
    f.close()
    direct = os.path.dirname(os.path.realpath(__file__)) + "\\dress_descriptions.txt"
    write_into_txt(descr_list, direct)
    l_mono = nlpcm.find_keys(directory=direct)
    l_bigram = list(nlpcm.find_keys(type_of_collocation="bigram_collocations", directory=direct).keys_counter)
    l_trigram = list(nlpcm.find_keys(type_of_collocation="trigram_collocations", directory=direct).keys_counter)
    l_bigram.sort(key=lambda tup: tup[1], reverse=True)
    l_trigram.sort(key=lambda tup: tup[1], reverse=True)
    f1 = open("mono.txt", "w")
    f2 = open("bi.txt", "w")
    f3 = open("tri.txt", "w")
    f1.writelines(str(l_mono))
    f2.writelines(str(l_bigram))
    f3.writelines(str(l_trigram))
    f1.close()
    f2.close()
    f3.close()


def main_function3():
    keys = [("Director", 1), ("Peter", 2), ("Jackson", 3), ("first", 4), ("came", 4),
            ("into", 4), ("contact", 9), ("with", 8), ('The', 1), ('Lord', 9), ('of', 8),
            ('Rings', 9), ('as', 9)]
    desc_list = [  # description_tuple("1", "Director Peter Jackson first came into contact with"),
                   # description_tuple("2","The Lord of the Rings as a new project, wondering "),
                   description_tuple("3", "The Jackson Lord of the Rings as a new project, wondering "),
                   description_tuple("4", "Director Peter Jackson first came into contact with")]
    attribute_extraction(keys, desc_list)


def function():
    print "function"

# f = open("bi.txt", "r")
# l = f.readlines()
# #l=l.split()
# print l
# print len(l)
# main_function3()
