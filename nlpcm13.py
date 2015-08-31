from __builtin__ import staticmethod

__author__ = 'sergey'

import os
import shutil
import pymongo
import html2text
import nltk
import collections
from nltk.collocations import *
import pickle
from copy import deepcopy

# -------SmartCounter structure-------
# Attributes:
#            * keys_counter: list of tuples (tuple[0] = word, tuple[1] = number of that word in text)
#            * txt_size: amount of all words in keys_counter list

SmartCounter = collections.namedtuple("SmartCounter", ['txt_size', 'keys_counter'])

# -------WordCount structure-------
# Attributes:
#            * key: word.
#            * set_class: class from which this word was taken.
#            * txt_size: amount of all words in keys_counter list.
#                        (amount of only "filtrated" words from text where the key was taken; text ~ class).
#            * keys_counter: amount of "key" words in the text.

WordCount = collections.namedtuple("wordCount", ['key', 'set_class', 'txt_size', 'keys_counter'])

# -------WordValue--------
# Attributes:
#            * key: word
#            * value: statistic number after computing of some statistical processes.

WordValue = collections.namedtuple("WordValue", ["key", "value"])

# -------WordRelation structure--------
# Attributes:
#            * primeW: word to which we will find related words from a single sentence.
#            * relatedWords: words that have got a relationship with prime word in a single sentence.
#                            relatedWords are presented as list of WordValue-s (named tuples).

WordRelation = collections.namedtuple("WordRelation", ["primeW", "relatedWords"])


class NLPCmatrix(object):
    # * NLPCmatrix (Natural Language Processing Classifier matrix) is a matrix which is used to hold
    #   an information about text.
    # * The main purpose of such a structure is to ease an implementation of "Natural Language Processes".
    # *attributes:
    #           1.matrix: matrix, each of its elements is a list.
    #           2.i_list: list of matrix's row "indexes" (keys)
    #           3.j_list: list of matrix's column "indexes" (keys)

    def __init__(self, i_rows, j_cols):
        """
        __init__ function - constructor of NLPCmatrix variable:
        creates dict matrix of empty lists with "i_rows" raws and "j_col" columns.
        :param i_rows: list of row keys.
        :param j_cols: list of column keys.
        :return: None
        """
        self.matrix = {}
        self.i_list = i_rows
        self.j_list = j_cols
        for i in i_rows:
            self.matrix[i] = {}
            for j in j_cols:
                self.matrix[i][j] = []

    def save(self, file_dir="save"):
        """
        Save function saves the current NLPCmatrix into file:
        :param file_dir: file directory (includes file name but without its type).
        :return: None
        """
        with open(file_dir + ".p", "wb") as f:
            for i in self.i_list:
                for j in self.j_list:
                    pickle.dump(self.matrix[i][j], f)
        f.close()

    def load(self, file_dir="save"):
        """
        Load function loads a content of already existed NLPCmatrix (previously saved) from file.
        WARNING: You may load only identical NLPCmatrix (i_list of keys and j_list of keys mast
        be the same in both NLPCmatrixes or at least their sizes).
        :param file_dir: file directory (includes file name but without its type).
        :return: None.
        """
        # If file in directory "file_dir" does not exist - raise error
        # otherwise load NLPCmatrix from the file.
        if os.path.exists(file_dir + ".p"):
            with open(file_dir + ".p", "rb") as f:
                for i in self.i_list:
                    for j in self.j_list:
                        self.matrix[i][j] = pickle.load(f)
            f.close()
        else:
            raise OSError.FileExistsError("s: file in directory " + file_dir + ".p" + " does not exist.")

    def apply_to_each_item(self, func, *arg):
        # TODO: check the work of this function + how to apply the obtainable func:
        """
        Each item of NLPCmatrix is a SmartCounter or list of SmartCounter-s.
        This function gives an opportunity to influence on each matrix element (only in case if
        this item is a SmartCounter structure or list of such structures).
        The function obtains function and its arguments. The obtainable function will be performed
        to each matrix's item.
        :param func: function which will process each NLPCmatrix element.
        :param arg: arguments that "func" function obtains.
        :return: None
        """
        # If current NLPCmatrix hasn't got columns or/and rows that means that this matrix is not
        # completed and cannot achieve the main purposes of NLPCmatrix class:
        if not self.i_list or not self.j_list:
            raise ValueError('s: wrong structure of NLPCmatrix variable')
        # If a matrix's item is SmartCounter:
        if type(self.matrix[self.i_list[0]][self.j_list[0]]) is SmartCounter:
            for i in self.i_list:
                for j in self.j_list:
                    func(self.matrix[i][j].keys_counter, *arg)
        # if a matrix's item is a list:
        elif type(self.matrix[self.i_list[0]][self.j_list[0]]) is list():
            for i in self.i_list:
                for j in self.j_list:
                    for list_item in self.matrix[i][j]:
                        # If a list's item is SmartCounter perform "func" function
                        # otherwise raise an error:
                        if type(list_item) is SmartCounter:
                            func(list_item.keys_counter, *arg)
                        else:
                            raise TypeError("s: NLPCmatrix variable mast contain only SmartCounter")
        else:
            raise TypeError("s: NLPCmatrix variable mast contain only SmartCounter")

    def print_matrix(self, reverse=False):
        """
        Prints NLPCmatrix
        :param reverse: reverse = True => print matrix[i][j]
                        reverse = False => print matrix[j][i]
        :return: None
        """
        if reverse:
            for j in self.j_list:
                print j + ":"
                print "---------------------"
                for i in self.i_list:
                    print i + ":"
                    print self.matrix[i][j]
        else:
            for i in self.i_list:
                print i + ":"
                print "---------------------"
                for j in self.j_list:
                    print j + ":"
                    print self.matrix[i][j]

    @staticmethod
    def read_from_file(directory):
        """
        The function returns a content of the file (only text).
        :param directory: string of directory to the file you want open.
        :return: (string) content of the file
        """
        with open(directory, 'r+')as f:
            row = f.read()
            f.close()
        return row

    @staticmethod
    def find_single_words(row=None, txt_file_dir=None):
        """
        The function obtains string (or directory to txt file from where it will receive
        string to work with )and returns SmartCounter structure:
        keys_counter is a list of tuples: tuple=(word, its amount in the string)
        this list contains only words presented in the obtainable string and
        suitable according to the filter function.
        IMPORTANT: Function works with only one argument!
        If function will obtain string and directory at one time => it will process only the
        obtainable string and not string saved under directory.
        :param row: string from which function will count words
        :param txt_file_dir: The directory to a txt file from where function will receive string to process
        :return: SmartCounter
        """
        if row is None and txt_file_dir is None:
            raise Exception('Error s0001: find_single_words function receives at least one parameter. \
                            No one parameter was given.')
        elif row is None and txt_file_dir is not None:
            row = NLPCmatrix.read_from_file(txt_file_dir)
        tokens = nltk.word_tokenize(row)
        text = nltk.Text(tokens)
        fDist = nltk.FreqDist(text)
        fDist = fDist.most_common(len(row))
        print fDist
        counter = [w for w in fDist if bool(NLPCmatrix.filter(w[0].lower()))]
        return SmartCounter(txt_size=sum(each_tuple[1] for each_tuple in counter), keys_counter=counter)

    @staticmethod
    def find_bigram_collocations(row=None, txt_file_dir=None):
        """
        The function obtains string (or directory to txt file from where it will receive
        string to work with )and returns SmartCounter structure:
        keys_counter is a list of tuples: tuple=((bigram collocation), its amount in the string)
        this list contains only collocations presented in the obtainable string and
        suitable according to the filter function.
        IMPORTANT: Function works with only one argument!
        If function will obtain string and directory at the time => it will process only the
        obtainable string and not string saved under directory.
        :param row: string from which function will count collocations
        :param txt_file_dir: The directory to a txt file from where function will receive string to process
        :return: SmartCounter
        """
        if row is None and txt_file_dir is None:
            raise Exception('Error s0001: find_bigram_collocations function receives at least one parameter. \
                            No one parameter was given.')
        elif row is not None:
            tokens = nltk.wordpunct_tokenize(row)
            finder = BigramCollocationFinder.from_words(tokens)
        elif txt_file_dir is not None:
            finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words(txt_file_dir))
        finder.apply_word_filter(lambda w: not NLPCmatrix.filter(w.lower()))
        counter = finder.ngram_fd.viewitems()
        return SmartCounter(txt_size=sum(each_tuple[1] for each_tuple in counter), keys_counter=counter)

    @staticmethod
    def find_trigram_collocations(row=None, txt_file_dir=None):
        """
        The function obtains string (or directory to txt file from where it will receive
        string to work with )and returns SmartCounter structure:
        keys_counter is a list of tuples: tuple=((trigram collocation), its amount in the string)
        this list contains only collocations presented in the obtainable string and
        suitable according to the filter function.
        IMPORTANT: Function works with only one argument!
        If function will obtain string and directory at one time => it will process only the
        obtainable string and not string saved under directory.
        :param row: string from which function will count collocations
        :param txt_file_dir: The directory to a txt file from where function will receive string to process
        :return: SmartCounter
        """
        if row is None and txt_file_dir is None:
            raise Exception('Error s0001: find_trigram_collocations function receives at least one parameter. \
                            No one parameter was given.')
        elif row is not None:
            tokens = nltk.wordpunct_tokenize(row)
            finder = TrigramCollocationFinder.from_words(tokens)
        elif txt_file_dir is not None:
            finder = TrigramCollocationFinder.from_words(nltk.corpus.genesis.words(txt_file_dir))
        finder.apply_word_filter(lambda w: not NLPCmatrix.filter(w.lower()))
        counter = finder.ngram_fd.viewitems()
        return SmartCounter(txt_size=sum(each_tuple[1] for each_tuple in counter), keys_counter=counter)

    @staticmethod
    def filter(word):
        """
        This function "decides" if the obtainable word is suitable to be a keyword.
        :param word: string
        :return: boolean value => True/False
        """
        result = True
        ignored_words = nltk.corpus.stopwords.words('english')
        result = bool(result and (len(word) > 1 and word.isalpha() and word not in ignored_words))
        return result

    @staticmethod
    def find_keys(source_string=None, type_of_collocation="single_keys", directory=None):
        """
        The function obtains source_string or/and directory (location from where to take the string)
        If function obtains these two variables simultaneously it will process only source_string (ignoring
        directory variable).
        Also find_keys obtains type_of_collocation variable (one of: "single_keys", "bigram_collocations",
        "trigram_collocations") and finds this collocations (or single words) in the obtainable string.
        Before return list of collocations function filters it from not necessary words by filter function.
        :param directory: (string) directory to string which will be processed by find_keys function.
        :param type_of_collocation: Type of collocation which the function will return.
        :param source_string: string which will be processed by find_keys function.
        :return: SmartCounter.
        """
        #   Find single keys:
        if type_of_collocation == "single_keys":
            return NLPCmatrix.find_single_words(source_string, directory)
        # Find bigram collocations:
        if type_of_collocation == "bigram_collocations":
            return NLPCmatrix.find_bigram_collocations(source_string, directory)
        # Find trigram collocations:
        if type_of_collocation == "trigram_collocations":
            return NLPCmatrix.find_trigram_collocations(source_string, directory)

    @staticmethod
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

    @staticmethod
    def get_all_dresses_data():
        """
        This function finds only data of dresses.
        :return: cursor to data of dresses (only id, categories and description)
        """
        db = pymongo.MongoClient().mydb
        category_id = "dresses"
        query = {"categories": {"$elemMatch": {"id": {"$in": \
                                                          NLPCmatrix.get_all_subcategories(db.categories,
                                                                                           category_id)}}}}
        fields = {"categories": 1, "id": 1, "description": 1}
        data = db.products.find(query, fields)
        return data

    @staticmethod
    def get_all_skirts_data():
        """
        This function finds only data of dresses.
        :return: cursor to data of dresses (only id, categories and description)
        """
        db = pymongo.MongoClient().mydb
        category_id = "skirts"
        query = {"categories": {"$elemMatch": {"id": {"$in": \
                                                          NLPCmatrix.get_all_subcategories(db.categories,
                                                                                           category_id)}}}}
        fields = {"categories": 1, "id": 1, "description": 1}
        data = db.products.find(query, fields)
        return data

    @staticmethod
    def get_all_data():
        """
        This function finds only data of dresses.
        :return: cursor to all data
        """
        db = pymongo.MongoClient().mydb
        return db.products.find()

    @staticmethod
    def html_to_str(h_str):
        """
        :param h_str: html code
        :return: a "clear" string without signs of html language.
        """
        str_m = html2text.html2text(h_str)
        str_m = str_m.encode('ascii', 'ignore')  # conversion of unicode type to string type
        return str_m

    @staticmethod
    def classify(main_dir, collocation_types_list, portion, data=None):
        """
        The function obtains directory to number of folders, each folder
        contains images which relate to a certain class (in our specific
        case - describes some type of clothing). All images in the folders
        also must exist in database, name of each image must be its id by
        which the function will find this image in database and therefore
        will have an opportunity to get its description.
        The function reads descriptions of all images per class (folder)
        and builds list of keys which relates to this class
        ("keys" - single words or collocations)
        :param main_dir: directory of folders with classified images.
        :param collocation_types_list: list of collocation types that function must find
        :param portion: Because of data's huge size we cannot process all descriptions at once,
                        therefore we define "portion number" (number of descriptions
                        that will be processed at one time - in another
                        words we will divide all descriptions into number of equals parts each
                        part contains "portion" number of descriptions, and process each one of them
                        separately).
        :param data: data (cursor). if function doesn't obtain a data variable => it will process
                     dresses data automatically.
        :return: NLPCmatrix
        """

        if data is None:
            # Find all dresses data (by default):
            data = NLPCmatrix.get_all_dresses_data()
        else:
            data = data
        # Find all classes' names:
        list_of_classes = os.listdir(main_dir)  # list of folder names.
        matrix = NLPCmatrix(list_of_classes, collocation_types_list)  # matrix - will contain all keys.
        matrix_of_id = {}  # matrix_of_id - per each class contains list of images names.
        list_of_files = {}
        # if "AllClasses" folder already exists => remove "AllClasses" folder:
        if os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/AllClasses"):
            shutil.rmtree("AllClasses")
        os.mkdir("AllClasses")
        # create and open txt files in "AllClasses" folder for each class from "list_of_classes" variable:
        for file_name in list_of_classes:
            f = open("AllClasses/" + file_name + ".txt", "w")
            f.close()
            f = open("AllClasses/" + file_name + ".txt", "a")
            # Fixate open files:
            list_of_files[file_name] = f
            # Per each class (folder name): save names of all images:
            matrix_of_id[file_name] = os.listdir(main_dir + "/" + file_name)
        length = data.count()
        count = 0
        recorded_count = 0
        dir = os.path.dirname(os.path.realpath(__file__)) + "/AllClasses/"
        for product in data:
            count += 1
            print count  # test
            # Perform "find_collocation" function if txt file has collected "portion"
            # number of descriptions or if txt file has collected the last data base's description
            if count % portion == 0 or count == length:
                for file_name in list_of_classes:
                    list_of_files[file_name].close()
                for file_name in list_of_classes:
                    for collocation_type in collocation_types_list:
                        matrix.matrix[file_name][collocation_type].append \
                            (NLPCmatrix.find_keys(type_of_collocation=collocation_type,
                                                  directory=dir + file_name + ".txt"))
                    with open('AllClasses/' + file_name + ".txt", "w")as f:
                        f.writelines(" 1 ")
                        f.close()
                    f = open('AllClasses/' + file_name + '.txt', 'a')
                    list_of_files[file_name] = f
            else:
                #  Take only description which relates to specific group:
                for file_name in list_of_classes:
                    if str(product["id"]) + ".jpg" in matrix_of_id[file_name]:
                        list_of_files[file_name].write(NLPCmatrix.html_to_str(product["description"]))
                        recorded_count += 1
        # After we have writen relevant descriptions into txt files we closes each of them
        #  and remove "AllClasses" folder with all its content (txt files with descriptions)
        for file_name in list_of_classes:
            list_of_files[file_name].close()
        shutil.rmtree("AllClasses")
        return matrix

    @staticmethod
    def union_of_lists(ortomatrix):
        """
        The function adds all SmartCounter-s, and if there after adding
        exist duplicated items function will add them too (adds number of their repetition in the list of SmartCounter).
        :param ortomatrix: list of SmartCounter-s.
        :return: sc_list without duplicated items.
        """
        united_list = []
        result = []
        count_source_text = 0
        #  Add lists:
        for sc_list in ortomatrix:
            united_list += sc_list.keys_counter
            count_source_text = sc_list.txt_size + count_source_text
        united_list.sort(key=lambda tup: tup[0], reverse=True)
        #  If united_list is not empty, do.
        if bool(united_list):
            item = united_list[0]
            # TODO: find a pythonic way to unite a duplicate items.
            #  Unite duplicate items:
            for list_item in united_list[1:]:
                if item[0] != list_item[0]:
                    result.append(item)
                    item = list_item
                else:
                    item = (item[0], list_item[1] + item[1])
            result.append(item)
            result.sort(key=lambda tup: tup[1], reverse=True)
        return SmartCounter(txt_size=count_source_text, keys_counter=result)

    def unite_dup_items(self):
        """
        Unite Duplicate Items.
        After an implementation of classify function NLPCmatrix contains duplicate SmartCounter-s.
        The purpose of the function to unite all duplicate SmartCount-s on each NLPCmatrix[i][j] list.
        :return: None
        """
        for i in self.i_list:
            for j in self.j_list:
                self.matrix[i][j] = NLPCmatrix.union_of_lists(self.matrix[i][j])

    @staticmethod
    def find_probability(word_tuple, list_probability):
        """
        The function calculates a probability to find word in one group and do not find in others
        :param word_tuple: tuple with word and its information
               (word, class_name, number_of_all_words_in_the_source_code, number_of_specific_word_in_the _source_text )
        :param list_probability: lists of tuples. Each tuple contains similar information as word_tuple,
                but with relation to its own, specific group.
        :return: float number => E (0;1)
        """
        probability = float(word_tuple.keys_counter) / float(word_tuple.txt_size)
        if len(list_probability) != 0:
            for tuple in list_probability:
                probability = probability * float(1 - (tuple.keys_counter) / float(tuple.txt_size))
        return probability

    def __fill_driver__(self, driver):
        """
        The function obtains NLPCmatrix (matrix each item of which is a SmartCounter, SmartCounter.keys_counter -
        sorted list of tuples (reversed = True); sorted by words).
        The main purpose of the function is to fill a driver matrix with minimum elements from the current matrix:
        for each item: driver.matrix[i][j]:=self.matrix[j],[i]
        and:
            self.i_list == driver.j_list
            self.j_list == driver.i_list
        :param driver: NLPCmatrix each item of which is an empty list (empty item) or WordCounter.
        :return:
        """
        for collocation_type in self.j_list:
            for class_name in self.i_list:
                #  Pop word from list only in case if it not empty and here is
                #  place for this item in temporary_dic!!!:.
                if len(self.matrix[class_name][collocation_type].keys_counter) != 0 \
                        and not driver.matrix[collocation_type][class_name]:
                    # Conversion of smart_tuple to word_tuple, because find_probability_function
                    # obtains only word_tuples
                    smart_tuple = self.matrix[class_name][collocation_type].keys_counter.pop()
                    driver.matrix[collocation_type][class_name] = \
                        WordCount(key=smart_tuple[0], set_class=class_name,
                                  txt_size=self.matrix[class_name][collocation_type].txt_size,
                                  keys_counter=smart_tuple[1])

    def evaluate(self):
        """
        evaluate function obtains NLPCmatrix (each matrix's item - SmartCounter,
        SmartCounter.keys_counter - list of tuples, each tuple in this list -
        (word, number of "word" in the text from where this SmartCounter was build),
        SmartCounter.txt_size - number of FILTRATED words in the text from which the SmartCounter was build).
        The function changes the structure of each item in the obtainable matrix:
        Item - list of tuples, each tuple - (word, value of probability to find the "word" in this specific
        class and not find it in others).
        :return: status - boolean value.
        """
        #  the purpose of the variable is to check an input of find probability function:
        status = True
        run = True
        # create a temporary matrix:
        temp_matrix = NLPCmatrix(deepcopy(list(self.i_list)), deepcopy(list(self.j_list)))
        # create a temporary dictionary
        temp_dict = NLPCmatrix(deepcopy(list(self.j_list)), deepcopy(list(self.i_list)))
        # sort each list in the obtainable matrix:
        self.apply_to_each_item(NLPCmatrix.l_sort, 0)
        while run:
            self.__fill_driver__(temp_dict)
            for collocation_t in self.j_list:
                evaluation_group = NLPCmatrix.get_keys_of_min_value_from_dict(temp_dict.matrix[collocation_t])
                #  if evaluation group of some type of collocation is empty =>
                #  delete this group from list_of_collocations
                #  that means we stop calculations with this group.
                if not evaluation_group:
                    self.j_list.remove(collocation_t)
                # Find probability to each word from evaluation_group list and save it into temporary matrix:
                for class_name1 in evaluation_group:
                    word_tuple_negative_list = []
                    word_tuple_positive = temp_dict.matrix[collocation_t][class_name1]
                    for class_name2 in evaluation_group:
                        if class_name1 != class_name2 and temp_dict.matrix[collocation_t][class_name2]:
                            word_tuple_negative_list.append(temp_dict.matrix[collocation_t][class_name2])
                    # if the elements of word_tuple_negative_list not equals, or word_tuple_positive is not
                    # equal to elements of word_tuple_negative_list:
                    if len(collections.Counter([tup.key for tup in word_tuple_negative_list] +
                                                       [word_tuple_positive.key])) != 1:
                        status = False
                    temp_matrix.matrix[class_name1][collocation_t].append( \
                        (word_tuple_positive.key,
                         NLPCmatrix.find_probability(word_tuple_positive, word_tuple_negative_list)))
                # Clear temporary dictionary from used items:
                for class_name in evaluation_group:
                    temp_dict.matrix[collocation_t][class_name] = ()
            # if list_of collocations is empty that means that we have stopped calculations of probabilities.
            run = bool(self.j_list)
        self.replace(temp_matrix)
        return status

    def replace(self, temp_matrix):
        """
        The function obtains current matrix and replaces its elements with elements of temp_matrix.
        WARNING: 1.replace function doesn't remove a temp_matrix => that means after an implementation of this function
                   we may access to the elements of current matrix also from temp_matrix.
                 2.current matrix and temp matrix must be equals:
                                                                 self.i_list = temp_matrix.i_list
                                                                 self.j_list = temp_matrix.j_list
        :param temp_matrix: NLPCmatrix
        :return:
        """
        self.i_list = temp_matrix.i_list
        self.j_list = temp_matrix.j_list
        for i in temp_matrix.i_list:
            for j in temp_matrix.j_list:
                self.matrix[i][j] = temp_matrix.matrix[i][j]

    @staticmethod
    def get_keys_of_min_value_from_dict(dict):
        """
        :param: dict of wort_tuples
                (word, class_name, number_of_all_words_in_the_source_code, number_of_specific_word_in_the _source_text )
        :return: List of dictionary keys of minimum values.
        """
        result = []
        # get only not None items:
        list_of_w_tuples = [w for w in dict.values() if w]
        if list_of_w_tuples:
            min_value = min(list_of_w_tuples, key=lambda tup: tup[0])
            for class_name in dict:
                if dict[class_name] and dict[class_name][0] == min_value[0]:
                    result.append(class_name)
        return result

    def __companion_words__(self, origin_word):
        """
        WARNING: the current matrix mast contain in self.j_list items: "bigram_collocation", "trigram_collocations",
        in self.matrix - matrix build from product descriptions that relates to origin_word by meaning.
        :param origin_word: string.
        :return:
        """
        result = NLPCmatrix(deepcopy(self.i_list), ["bigram_collocations", "trigram_collocations"])
        for class_name in self.i_list:
            for collocation_name in result.j_list:
                for w_tuple in self.matrix[class_name][collocation_name].keys_counter:
                    if origin_word in str(w_tuple):
                        result.matrix[class_name][collocation_name].append(deepcopy(w_tuple))
        result.print_matrix()
        return result

    @staticmethod
    def create_txt_data(products_data, description_key):
        """
        The function obtains cursor to a data and key by which we will find description from each
        product in the data.
        The function buil
        """



        # def (main_dir, collocation_types_list, portion, data=None):
        #     """
        #     The function obtains directory to number of folders, each folder
        #     contains images which relate to a certain class (in our specific
        #     case - describes some type of clothing). All images in the folders
        #     also must exist in database, name of each image must be its id by
        #     which the function will find this image in database and therefore
        #     will have an opportunity to get its description.
        #     The function reads descriptions of all images per class (folder)
        #     and builds list of keys which relates to this class
        #     ("keys" - single words or collocations)
        #     :param main_dir: directory of folders with classified images.
        #     :param collocation_types_list: list of collocation types that function must find
        #     :param portion: Because of data's huge size we cannot process all descriptions at once,
        #                     therefore we define "portion number" (number of descriptions
        #                     that will be processed at one time - in another
        #                     words we will divide all descriptions into number of equals parts each
        #                     part contains "portion" number of descriptions, and process each one of them
        #                     separately).
        #     :param data: data (cursor). if function doesn't obtain a data variable => it will process
        #                  dresses data automatically.
        #     :return: NLPCmatrix
        #     """
        #
        #     if data is None:
        #         # Find all dresses data (by default):
        #         data = NLPCmatrix.get_all_dresses_data()
        #     else:
        #         data = data
        #     # Find all classes' names:
        #     list_of_classes = os.listdir(main_dir)  # list of folder names.
        #     matrix = NLPCmatrix(list_of_classes, collocation_types_list)  # matrix - will contain all keys.
        #     matrix_of_id = {}  # matrix_of_id - per each class contains list of images names.
        #     list_of_files = {}
        #     # if "AllClasses" folder already exists => remove "AllClasses" folder:
        #     if os.path.exists(os.path.dirname(os.path.realpath(__file__))+"/AllClasses"):
        #         shutil.rmtree("AllClasses")
        #     os.mkdir("AllClasses")
        #     # create and open txt files in "AllClasses" folder for each class from "list_of_classes" variable:
        #     for file_name in list_of_classes:
        #         f = open("AllClasses/"+file_name+".txt", "w")
        #         f.close()
        #         f = open("AllClasses/"+file_name+".txt","a")
        #         # Fixate open files:
        #         list_of_files[file_name] = f
        #         # Per each class (folder name): save names of all images:
        #         matrix_of_id[file_name] = os.listdir(main_dir+"/"+file_name)
        #     length = data.count()
        #     count = 0
        #     recorded_count = 0
        #     dir = os.path.dirname(os.path.realpath(__file__))+"/AllClasses/"
        #     for product in data:
        #         count += 1
        #         print count  # test
        #         # Perform "find_collocation" function if txt file has collected "portion"
        #         # number of descriptions or if txt file has collected the last data base's description
        #         if count % portion == 0 or count == length:
        #             for file_name in list_of_classes:
        #                 list_of_files[file_name].close()
        #             for file_name in list_of_classes:
        #                 for collocation_type in collocation_types_list:
        #                     matrix.matrix[file_name][collocation_type].append\
        #                         (NLPCmatrix.find_keys(type_of_collocation=collocation_type, directory=dir+file_name+".txt"))
        #                 with open('AllClasses/'+file_name+".txt", "w")as f:
        #                     f.writelines(" 1 ")
        #                     f.close()
        #                 f = open('AllClasses/'+file_name+'.txt', 'a')
        #                 list_of_files[file_name] = f
        #         else:
        #             #  Take only description which relates to specific group:
        #             for file_name in list_of_classes:
        #                 if str(product["id"])+".jpg" in matrix_of_id[file_name]:
        #                     list_of_files[file_name].write(NLPCmatrix.html_to_str(product["description"]))
        #                     recorded_count += 1
        #     #  After we have writen relevant descriptions into txt files we closes each of them
        #     #  and remove "AllClasses" folder with all its content (txt files with descriptions)
        #     for file_name in list_of_classes:
        #         list_of_files[file_name].close()
        #     shutil.rmtree("AllClasses")
        #     return matrix

    # ---------------secondary functions------------------:
    @staticmethod
    def l_sort(sc_list, item_num):
        """
        This function is a version of list.sort() function but portable to using
        by apply_to_each_item function.
        :param sc_list: list of tuples
        :param item_num: place of item in the list's tuple by which the sorting will be performed.
        :return: None
        """
        sc_list.sort(key=lambda tup: tup[item_num], reverse=True)

# #------------check save, load -----------
# matrix = NLPCmatrix(["maxi", "midi", "mini", "mini-1"], ["single_keys", "bigram_collocations", "trigram_collocations"])
# matrix.load()
# #matrix.apply_to_each_item(NLPCmatrix.l_sort, 1)
# #matrix.evaluate()
# #print matrix.matrix["maxi"]["single_keys"]
# #matrix.print_matrix(True)
# run = True
#
# #
# # run = True
# # print "start"
# # while run:
# #     print "write word:"
# #     print "for exit paste -1"
# #     word = input()
# #     if word == -1:
# #         run = False
# #     syns = matrix.__companion_words__(str(word))
# #     syns.print_matrix()
# #     plt.plot([tup[1] for tup in syns.matrix["maxi"]["bigram_collocations"]])
# #     plt.ylabel('some numbers')
# #     plt.show()
#
# # new_matrix = NLPCmatrix.classify("C:\Users\sergey\Desktop\skirts",
# #                                  ["single_keys", "bigram_collocations", "trigram_collocations"],
# #                                  10000, NLPCmatrix.get_all_skirts_data())
# # new_matrix.unite_dup_items()
# # new_matrix.save("skirts")
#
# #new_matrix = NLPCmatrix(["long-skirts","mid-length-skirts","mini-skirts"], ["single_keys", "bigram_collocations", "trigram_collocations"])
# data = NLPCmatrix.get_all_data()
# new_matrix = NLPCmatrix.classify("C:\Users\sergey\Desktop\skirts", ["long-skirts", "mid-length-skirts", "mini-skirts"], 5000, data)
# new_matrix.save("skirts")
# # new_matrix.load("skirts")
# new_matrix.print_matrix()
# #
# # while run:
# #     print "write word:"
# #     print "for exit paste -1"
# #     word = input()
# #     if word == -1:
# #         run = False
# #     syns = new_matrix.__companion_words__(str(word))
# #     syns.print_matrix()
# #     plt.plot([tup[1] for tup in syns.matrix["long-skirts"]["bigram_collocations"]])
# #     plt.ylabel('some numbers')
# #     plt.show()
# #new_mat = NLPCmatrix.classify("C:\Users\sergey\Desktop\skirts", ["long-skirts","mid-length-skirts","mini-skirts"], 5000)
# #new_mat.save("skirts")
# m = NLPCmatrix
