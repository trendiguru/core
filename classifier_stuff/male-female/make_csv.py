__author__ = 'jeremy'

import os.path
import shutil

import cv2


def show_files():
    BASE_PATH = os.getcwd()
    BASE_PATH = os.path.join(BASE_PATH, 'female')
    print('basepath:' + BASE_PATH)
    males = []
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        dirnames.sort()
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print('path:' + abs_path)
                img_arr = cv2.imread(abs_path)
                if img_arr is None:
                    continue
                cv2.imshow('file', img_arr)
                cv2.waitKey(1)


def sort_the_unsorted():
    BASE_PATH = os.getcwd()
    BASE_PATH2 = os.path.join(BASE_PATH, 'unknown')
    print('basepath:' + BASE_PATH2)
    males = []
    for dirname, dirnames, filenames in os.walk(BASE_PATH2):
        dirnames.sort()
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print('path:' + abs_path)
                img_arr = cv2.imread(abs_path)
                if img_arr is None:
                    continue
                cv2.imshow('file', img_arr)
                cv2.waitKey(1)
                a = raw_input('m for male and f for female PLEASE')
                src = abs_path
                if a == 'm' or a == 'M':
                    dst = os.path.join(BASE_PATH, 'test/male/' + filename)
                    print('moving ' + src + ' to ' + dst)
                    shutil.move(src, dst)
                elif a == 'f' or a == 'F':
                    dst = os.path.join(BASE_PATH, 'test/female/' + filename)
                    print('moving ' + src + ' to ' + dst)
                    shutil.move(src, dst)


if __name__ == "__main__":

    # if len(sys.argv) != 2:
    #        print "usage: create_csv <base_path>"
    #        sys.exit(1)

    #    BASE_PATH=sys.argv[1]
    if (1):
        sort_the_unsorted()
    # show_files()

    csv_filename = 'genders.csv'
    SEPARATOR = ";"
    with open(csv_filename, 'w') as f:
        #WOMEN
        f2 = open('women' + csv_filename, 'w')
        BASE_PATH = os.getcwd()
        BASE_PATH = os.path.join(BASE_PATH, 'female')
        print('basepath:' + BASE_PATH)
        label = 1
        for dirname, dirnames, filenames in os.walk(BASE_PATH):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    abs_path = "%s/%s" % (subject_path, filename)
                    s = "{0}{1}{2}\n".format(abs_path, SEPARATOR, label)
                    print(str(s))
                    f.write(s)
                    f2.write(s)
                    #MEN
        f2.close()
        f2 = open('men' + csv_filename, 'w')
        BASE_PATH = os.getcwd()
        BASE_PATH = os.path.join(BASE_PATH, 'male')
        print('basepath:' + BASE_PATH)
        label = 0
        for dirname, dirnames, filenames in os.walk(BASE_PATH):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    abs_path = "%s/%s" % (subject_path, filename)
                    s = "{0}{1}{2}\n".format(abs_path, SEPARATOR, label)
                    print(str(s))
                    f.write(s)
                    f2.write(s)
