__author__ = 'jeremy'

import os.path


if __name__ == "__main__":

    # if len(sys.argv) != 2:
    #        print "usage: create_csv <base_path>"
    #        sys.exit(1)

    #    BASE_PATH=sys.argv[1]

    csv_filename = 'genders.csv'
    SEPARATOR = ";"
    with open(csv_filename, 'w') as f:
        #WOMEN
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
                    #MEN
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