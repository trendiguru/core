import cv2
import urllib
import pymongo
import os
import urlparse
#import default
#import find_similar_mongo
import unittest
import imp
import sys
import pymongo
import fingerprint_core as fp
import Utils
import NNSearch
import numpy as np

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
        print('looking for string:'+str(strN))
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


def lookfor_next():
    print('path='+str(sys.path))
    resultDict = {}  #return empty dict if no results found
    prefixes = ['Main Image URL angle ', 'Style Gallery Image ']

    doc = next(training_collection_cursor, None)
    while doc is not None:
        print('doc:'+str(doc))
        tot_answers=[]
        for prefix in prefixes:
            answers = lookfor_image_group(doc, prefix)
            if answers is not None:
                tot_answers.append[answers]
        print('result:'+str(tot_answers))

def compare_fingerprints(image_array1,image_array2):
    good_results=[]
    power = 1.5
    tot_dist = 0 
    n = 0
    i = 0
    j = 0
    use_visual_output = False
    use_visual_output2 = False
    for entry1 in image_array1:
	print('entry1:'+str(entry1))
    	bb1 = entry1['human_bb']
    	url1 = entry1['url']
	print url1
   	img_arr1 = Utils.get_cv2_img_array(url1)
    	if img_arr1 is not None:
		fp1 = fp.fp(img_arr1,bb1)
		print('fp1:'+str(fp1))
  		i = i +1
 		j = 0
		if use_visual_output:
			cv2.imshow('im1',img_arr1)
 			k=cv2.waitKey(50)& 0xFF
    		for entry2 in image_array2:
			print('entry2:'+str(entry2))
    			bb2 = entry2['human_bb']
    			url2 = entry2['url']
			print url2
	   		img_arr2 = Utils.get_cv2_img_array(url2)
			if img_arr2 is not None:
				if use_visual_output2:
					cv2.imshow('im2',img_arr2)
		 			k=cv2.waitKey(50) & 0xFF
				j = j + 1
    				fp2 = fp.fp(img_arr2,bb2)
				#print('fp2:'+str(fp2))
    				dist = NNSearch.distance_1_k(fp1, fp2,power)
				tot_dist=tot_dist+dist
				print('distance:'+str(dist)+' totdist:'+str(tot_dist)+' comparing images '+str(i)+','+str(j))
				n=n+1
			else:
				print('bad img array 2')
	else:
		print('bad img array 1')

    avg_dist = float(tot_dist)/float(n)
    print('average distance:'+str(avg_dist)+',n='+str(n)+',tot='+str(tot_dist))
    return(avg_dist)

def normalize_matrix(matrix):
	# the matrix should be square and is only populated in top triangle , including the diagonal
	# so the number of elements is 1+2+...+N  for an  NxN array, which comes to N*(N+1)/2
    n_elements =float(matrix.shape[0]*matrix.shape[0]+matrix.shape[0])/2.0   
    sum = np.sum(matrix)
    avg = sum / n_elements
    normalized_matrix = np.divide(matrix,avg)
    return(normalized_matrix)

def cross_compare(image_sets):
    confusion_matrix = np.zeros((len(image_sets),len(image_sets)))
    print('confusion matrix size:'+str(len(image_sets))+' square')
    for i in range(0,len(image_sets)):
    	for j in range(i,len(image_sets)):
		print('comparing group '+str(i)+' to group '+str(j))
		print('group 1:'+str(image_sets[i]))
		print('group 2:'+str(image_sets[j]))
    		avg_dist = compare_fingerprints(image_sets[i],image_sets[j])
		confusion_matrix[i,j]=avg_dist
		print('confusion matrix is currently:'+str(confusion_matrix))
    normalized_matrix = normalize_matrix(confusion_matrix)
    return(normalized_matrix)

def mytrace(matrix):
    sum=0
    for i in range(0,matrix.shape[0]):
    	sum=sum+matrix[i,i]
    return(sum)

def calculate_normalized_confusion_matrix():
    db=pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()   #The db with multiple figs of same item
    assert(training_collection_cursor)  #make sure training collection exists
    doc = next(training_collection_cursor, None)
    i=0
    tot_answers=[]
    while doc is not None and i<3:
#        print('doc:'+str(doc))
        if doc["images"] is not None:
            tot_answers.append(doc['images'])
#    	    print('result:'+str(answers))
	    i=i+1
    	doc = next(training_collection_cursor, None)
    print('tot number of groups:'+str(i)+'='+str(len(tot_answers)))
    print('tot_answers:'+str(tot_answers))
    confusion_matrix = cross_compare(tot_answers)
    print('confusion matrix:'+str(confusion_matrix))
    return(confusion_matrix) 

def rate_fingerprint():
    confusion_matrix = calculate_normalized_confusion_matrix()
    #number of diagonal and offdiagonal elements for NxN array  is N and (N*N-1)/2
    n_diagonal_elements = confusion_matrix.shape[0]
    n_offdiagonal_elements = float(confusion_matrix.shape[0]*confusion_matrix.shape[0]-confusion_matrix.shape[0])/2.0
    same_item_avg = mytrace(confusion_matrix)/n_diagonal_elements
    different_item_avg = (float(np.sum(confusion_matrix))-float(mytrace(confusion_matrix))) / n_offdiagonal_elements
    goodness = different_item_avg - same_item_avg
    print('same item average:'+str(same_item_avg)+' different item average:'+str(different_item_avg)+' difference:'+str(goodness))
    return(goodness)


if __name__ == '__main__':
    rate_fingerprint()


