__author__ = 'jeremy'


import csv
import os

def read_csv(filename='/data/olympics/olympicsfull.csv'):
    filename = "olympicsfull.csv"
    unique_descs=[]
    with open(filename, "rb") as file:
        reader = csv.DictReader(file)
        for row in reader:
            print row
            if not row['description'] in unique_descs:
                unique_descs.append(row['description'])
                print unique_descs

def make_rcnn_trainfile(dir,filter='.jpg',trainfile='train.txt'):
    '''
    https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train
    better yet ssee https://github.com/deboc/py-faster-rcnn/tree/master/help
    
    :param dir:
    :param filter:
    :param trainfile:
    :return:
    '''
    files = [f for f in os.listdir(dir) if filter in f]
    with open(trainfile,'w') as fp:
        for f in files:
            stripped = f.replace('.jpg','')
            fp.write(stripped+'\n')
        fp.close()


	    # Do awesome things with row["path"], row["boundingBoxX"], etc..."
		# DictReader autommatically turn the row into a dict.

