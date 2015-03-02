import cv2
import urllib
import pymongo
import os
import urlparse
#import default
import find_similar_mongo
import unittest
import imp
import sys
import pymongo


#    def tear_down(self):
#        shutil.rmtree(self.temp_dir)

#this is for the training collection, where there's a set of images from different angles in each record
def lookfor_image_group(queryobject,string):
    n=1
    urlN=None   #if nothing eventually is found None is returned for url
    answer_url_list=[]
    bb=None
    while(1):
        theBB=None
        strN=string+str(n)  #this is to build strings like 'Main Image URL angle 5' or 'Style Gallery Image 7'
#        bbN = strN+' bb' #this builds strings like 'Main Image URL angle 5 bb' or 'Style Gallery Image 7 bb'
        print('looking for string:'+str(strN)+' and bb '+str(bbN))
    #	logging.debug('looking for string:'+str(strN)+' and bb '+str(bbN))
        if strN in queryobject:
            urlN=queryobject[strN]
            if not 'human_bb' in queryobject:  # got a pic without a bb
                got_unbounded_image = True
                print('image from string:'+strN+' :is not bounded!!')
            elif queryobject['human_bb'] is None:
                got_unbounded_image = True
                print('image from string:'+strN+' :is not bounded!!')
            else:
                got_unbounded_image = False
                print('image from string:'+strN+' :is bounded :(')
                theBB = queryobject['human_bb']
            current_answer = {'url':urlN,'bb':theBB}
            answer_url_list.append(current_answer)
            print('current answer:'+str(current_answer))
        else:
            print('didn\'t find expected string in training db')
            break
    return(answer_url_list)
# maybe return(urlN,n) at some point



def lookfor_next(self):

    print('path='+str(sys.path))
    resultDict = {}  #return empty dict if no results found
    prefixes = ['Main Image URL angle ', 'Style Gallery Image ']

    doc = next(self.training_collection_cursor, None)
    while doc is not None:
        print('doc:'+str(doc))
        tot_answers=[]
        for prefix in prefixes:
            answers = lookfor_image_group(doc, prefix)
            if answers is not None:
                tot_answers.append[answers]
        print('result:'+str(tot_answers))


if __name__ == '__main__':

    db=pymongo.MongoClient().mydb
    self.training_collection_cursor = db.training.find()   #The db with multiple figs of same item
    assert(self.training_collection_cursor)  #make sure training collection exists
    lookfor_next()


