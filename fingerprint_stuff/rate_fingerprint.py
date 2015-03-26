#todo weight averages by number of pics
#compute stdev and add to report
#done: fix ConnectionError: HTTPConnectionPool(host='img.sheinside.com', port=80): Max retries exceeded with url: /images/lookbook/wearing/201428/04181405101082542276157.jpg (Caused by <class 'socket.error'>: [Errno 104] Connection reset by peer)
#TODO make sure fp is correct when image is missing/not available (make sure its not counted)
#from joblib import Parallel, delayed
#NOTE - cross-compare not yet implementing weights, fp_function,distance_function,distance_power
from __future__ import print_function
from multiprocessing import Pool
import datetime
import json
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
import fingerprint_core as fp_core
import Utils
import NNSearch
import numpy as np
import cProfile
import StringIO
import pstats
import  background_removal
import pdb 
import logging

min_images_per_doc = 10
max_items = 100

BLUE = [255, 0, 0]        # rectangle color
RED = [0, 0, 255]         # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

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

def lookfor_next_imageset():
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

def compare_fingerprints(image_array1,image_array2,fingerprint_function=fp_core.fp,weights=np.ones(fingerprint_length),distance_function=NNSearch.distance_1_k,distance_power=1.5):
    good_results=[]
    tot_dist = 0 
    n = 0
    i = 0
    j = 0
    use_visual_output = False
    use_visual_output2 = False
    for entry1 in image_array1:
    	bb1 = entry1['human_bb']
    	url1 = entry1['url']
   	img_arr1 = Utils.get_cv2_img_array(url1,try_url_locally=True,download=True)
    	if img_arr1 is not None:
		fp1 = fp.fingerprint_function(img_arr1,bounding_box=bb1,weights=weights)
	#	print('fp1:'+str(fp1))
  		i = i +1
#		print('image '+str(i)+':'+str(entry1))
 		j = 0
		if use_visual_output:
			cv2.rectangle(img_arr1, (bb1[0],bb1[1]), (bb1[0]+bb1[2], bb1[1]+bb1[3]), color = GREEN,thickness=2)
			cv2.imshow('im1',img_arr1)
 			k=cv2.waitKey(50)& 0xFF
#to parallelize
#[sqrt(i ** 2) for i in range(10)]
#Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    		for entry2 in image_array2:
    			bb2 = entry2['human_bb']
    			url2 = entry2['url']
	   		img_arr2 = Utils.get_cv2_img_array(url2,try_url_locally=True,download=True)
			if img_arr2 is not None:
				if use_visual_output2:
 					cv2.rectangle(img_arr2, (bb2[0],bb2[1]), (bb2[0]+bb2[2], bb2[1]+bb2[3]), color=BLUE,thickness=2)
					cv2.imshow('im2',img_arr2)
		 			k=cv2.waitKey(50) & 0xFF
				j = j + 1
#				print('image '+str(j)+':'+str(entry2))
    				fp2 = fp.fingerprint_function(img_arr2,bounding_box=bb2,weights=weights)
				#print('fp2:'+str(fp2))
				#pdb.set_trace()
    				dist = NNSearch.distance_function(fp1, fp2,distance_power)
				tot_dist=tot_dist+dist
				print('distance:'+str(dist)+' totdist:'+str(tot_dist)+' when comparing images '+str(i)+','+str(j),end='\r',sep='')
				n=n+1
			else:
				print('bad img array 2')
	else:
		print('bad img array 1')

    avg_dist = float(tot_dist)/float(n)
    print('average distance:'+str(avg_dist)+',n='+str(n)+',tot='+str(tot_dist))
    return(avg_dist)

def compare_fingerprints_except_diagonal(image_array1,image_array2,fingerprint_function=fp_core.fp,weights=np.ones(fingerprint_length),distance_function=NNSearch.distance_1_k,distance_power=1.5):
#    assert(len(image_array1) == len(image_array2)) #maybe not require that these be the same set...
#    print('fp_func:'+str(fingerprint_function))
#    print('weights:'+str(weights))
#    print('distance_function:'+str(distance_function))
#    print('distance_power:'+str(distance_power))
    good_results=[]
    tot_dist = 0 
    n = 0
    i = 0
    j = 0
    use_visual_output = False
    use_visual_output2 = False
    distance_array=[]
    for entry1 in image_array1:
  	i = i +1
#	print('image 1:'+str(entry1))
    	bb1 = entry1['human_bb']
    	url1 = entry1['url']
   	img_arr1 = Utils.get_cv2_img_array(url1,try_url_locally=True,download=True)
    	if img_arr1 is not None:
  		#background_removal.standard_resize(image, 400) 
		fp1 = fingerprint_function(img_arr1,bounding_box=bb1,weights=weights)
#		print('fp1:'+str(fp1))
 		j = 0
		if use_visual_output:
			cv2.rectangle(img_arr1, (bb1[0],bb1[1]), (bb1[0]+bb1[2], bb1[1]+bb1[3]), color = GREEN,thickness=2)
			cv2.imshow('im1',img_arr1)
 			k=cv2.waitKey(50)& 0xFF
#to parallelize
#[sqrt(i ** 2) for i in range(10)]
#Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    		for entry2 in image_array2:
			j = j + 1
#			print('image 2:'+str(entry2))
    			bb2 = entry2['human_bb']
    			url2 = entry2['url']
	   		img_arr2 = Utils.get_cv2_img_array(url2,try_url_locally=True,download=True)
			if img_arr2 is not None:
				if use_visual_output2:
 					cv2.rectangle(img_arr2, (bb2[0],bb2[1]), (bb2[0]+bb2[2], bb2[1]+bb2[3]), color=BLUE,thickness=2)
					cv2.imshow('im2',img_arr2)
		 			k=cv2.waitKey(50) & 0xFF
				#pdb.set_trace()
    				fp2 = fingerprint_function(img_arr2,bounding_box=bb2,weights=weights)
				#print('fp2:'+str(fp2))
    				dist = distance_function(fp1, fp2,k=distance_power)
				if i != j: 
					distance_array.append(dist)
				tot_dist=tot_dist+dist
				print('distance:'+str(dist)+' totdist:'+str(tot_dist)+' comparing images '+str(i)+','+str(j)+'      ',end='\r',sep='')
				n=n+1
			else:
				print('bad img array 2')
				logging.debug('bad image array 1 in rate_fingerprint.py:compare_fignreprints_ecept_diagonal')
	else:
		print('bad img array 1')
		logging.debug('bad image array 1 in rate_fingerprint.py:compare_fignreprints_ecept_diagonal')
    n_diagonal_elements = i
    avg_dist = float(tot_dist)/float(n-i)  #this is the one part thats different between compare_fp_except_diagonal and compare_fp
    distances_np_array = np.array(distance_array)
    distances_stdev = np.std(distances_np_array)
    distances_mean = np.mean(distances_np_array)
    print('average distance:'+str(avg_dist)+',stdev'+str(distances_stdev)+',n='+str(n)+',tot='+str(tot_dist)+' diag elements:'+str(i))
#    print('average distance numpy:'+str(distances_mean)+',stdev'+str(distances_stdev))
    return(avg_dist,distances_stdev)

def normalize_matrix(matrix):
	# the matrix should be square and is only populated in top triangle , including the diagonal
	# so the number of elements is 1+2+...+N  for an  NxN array, which comes to N*(N+1)/2
    n_elements =float(matrix.shape[0]*matrix.shape[0]+matrix.shape[0])/2.0   
    sum = np.sum(matrix)
    avg = sum / n_elements
    normalized_matrix = np.divide(matrix,avg)
    return(normalized_matrix)

def cross_compare(image_sets):
    '''
    compares image set i to image set j (including j=i)
    '''
    confusion_matrix = np.zeros((len(image_sets),len(image_sets)))
    print('confusion matrix size:'+str(len(image_sets))+' square')
    for i in range(0,len(image_sets)):
    	for j in range(i,len(image_sets)):
		print('comparing group '+str(i)+' to group '+str(j))
		print('group 1:'+str(image_sets[i]))
		print('group 2:'+str(image_sets[j]))
		if (i==j):
	    		avg_dist = compare_fingerprints_except_diagonal(image_sets[i],image_sets[j])
		else:
	    		avg_dist = compare_fingerprints(image_sets[i],image_sets[j])
		confusion_matrix[i,j]=avg_dist
		print('confusion matrix is currently:'+str(confusion_matrix))
#    normalized_matrix = normalize_matrix(confusion_matrix)
#    return(normalized_matrix)
    return(confusion_matrix)

def self_compare(image_sets,fingerprint_function=fp_core.fp,weights=np.ones(fingerprint_length),distance_function=NNSearch.distance_1_k,distance_power=1.5):
    '''
    compares image set i to image set i
    '''
    confusion_matrix = np.zeros((len(image_sets)))
    stdev_matrix = np.zeros((len(image_sets)))
#    print('confusion vector size:'+str(len(image_sets))+' long')
    for i in range(0,len(image_sets)):
	print('comparing group '+str(i)+' to itself')
#	print('group '+str(i)+':'+str(image_sets[i]))
    	avg_dist,stdev = compare_fingerprints_except_diagonal(image_sets[i],image_sets[i],fingerprint_function=fingerprint_function,weights=weights,distance_function=distance_function,distance_power=distance_power)
	confusion_matrix[i] = avg_dist
	stdev_matrix[i] = stdev
#	print('confusion vector is currently:'+str(confusion_matrix))
#    normalized_matrix = normalize_matrix(confusion_matrix)
#    return(normalized_matrix)
    return(confusion_matrix,stdev_matrix)

def mytrace(matrix):
    sum=0
    for i in range(0,matrix.shape[0]):
    	sum=sum+matrix[i,i]
    return(sum)

def calculate_confusion_matrix():
    global report
    report = {'n_groups':0,'n_items':[],'confusion_matrix':[]}
    min_images_per_doc = 5
    db=pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()   #The db with multiple figs of same item
    assert(training_collection_cursor)  #make sure training collection exists
    doc = next(training_collection_cursor, None)
    i=0
    tot_answers=[]
    while doc is not None and i<max_items:   #just take 1st N for testing
#        print('doc:'+str(doc))
        images = doc['images']
        n_images = len(images)
        n_good = Utils.count_human_bbs_in_doc(images)
        if n_good > min_images_per_doc:
        	i = i + 1
                print('got '+str(n_good)+' bounded images, '+str(min_images_per_doc)+' required, '+str(n_images)+' images tot');
                tot_answers.append(get_images_from_doc(images))
                report['n_items'].append(n_good)
        else:
                print('not enough bounded boxes (only '+str(n_good)+' found, of '+str(min_images_per_doc)+' required, '+str(n_images)+' images tot',end='\r',sep='')
    	doc = next(training_collection_cursor, None)
    print('tot number of groups:'+str(i)+'='+str(len(tot_answers)))
    confusion_matrix = cross_compare(tot_answers)
    print('confusion matrix:'+str(confusion_matrix))
    report['confusion_matrix'] = confusion_matrix.tolist() #this is required for json dumping
#    report['fingerprint_function']='fp'
    report['distance_function'] = 'NNSearch.distance_1_k(fp1, fp2,power=1.5)'
    report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
#    save_report(report)
    return(confusion_matrix) 

def get_images_from_doc(images):
    '''
    return the good (bounded) images from an images doc
    '''
    pruned_images = []
    for img in images:
    	if Utils.good_bb(img):
		pruned_images.append(img)
#    print('pruned images:')
#    nice_print(pruned_images)
    return(pruned_images)

def nice_print(images):
    i=1
    for img in images:
	print('img '+str(i)+':'+str(img))
	i=i+1


def calculate_self_confusion_vector(fingerprint_function=fp_core.fp,weights=np.ones(fingerprint_length),distance_function=NNSearch.distance_1_k,distance_power=1.5):
    #don't look at sets with less than this number of images
    global report
    db=pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()
    assert(training_collection_cursor)  #make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    tot_answers=[]
    report = {'n_groups':0,'n_images':[]}
    while doc is not None and i<max_items:
#        print('doc:'+str(doc))
	images = doc['images']
        if images is not None:
		n_images = len(images)
		n_good = Utils.count_human_bbs_in_doc(images)
            	if n_good > min_images_per_doc:
			i = i + 1
			print('got '+str(n_good)+' bounded images, '+str(min_images_per_doc)+' required, '+str(n_images)+' images tot')
			tot_answers.append(get_images_from_doc(images))
			report['n_images'].append(n_good)
		else:
			print('not enough bounded boxes (only '+str(n_good)+' found, of '+str(min_images_per_doc)+' required, '+str(n_images)+' images tot',end='\r',sep='')
   	doc = next(training_collection_cursor, None)
    confusion_vector,stdev_vector = self_compare(tot_answers,fingerprint_function=fingerprint_function,weights=weights,distance_function=distance_function,distance_power=distance_power)
    print('tot number of groups:'+str(i)+'='+str(len(tot_answers)))
    print('confusion vector:'+str(confusion_vector))
    report['n_groups'] = i
    report['confusion_vector'] = confusion_vector.tolist() #this is required for json dumping
    report['error_vector'] = stdev_vector.tolist() #this is required for json dumping
 #   report['fingerprint_function']=fingerprint_function
    report['weights'] = weights.tolist()
    #report['distance_function'] = distance_function
    report['distance_power'] = distance_power
    report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    #save_report(report)
    weighted_average = 0
    tot_images = 0
    cumulative_error = 0
    for j in range(0,i):
	weighted_average = weighted_average + report['n_images'][j]*confusion_vector[j]
	tot_images = tot_images + report['n_images'][j]
	cumulative_error = cumulative_error + (report['n_images'][j]*stdev_vector[j])*(report['n_images'][j]*stdev_vector[j]) #error adds in quadrature
	print('error element:'+str((report['n_images'][j]*stdev_vector[j])*(report['n_images'][j]*stdev_vector[j])))
    weighted_average = weighted_average/tot_images
    cumulative_error = np.sqrt(cumulative_error)/tot_images 
    print('weighted_average:'+str(weighted_average))
    report['average_weighted'] = weighted_average
    print('cumulative error:'+str(cumulative_error))
    report['error_cumulative'] = cumulative_error
    return(confusion_vector) 

def save_report(report):
    try:
	f = open('fp_ratings.txt', 'a')  #ha!! mode 'w+' .... overwrites the file!!!
    except IOError:
        print ('cannot open fp_ratings.txt')
    else:
    	print('reporting...'+str(report))
    	json.dump(report,f,indent=4,sort_keys=True,separators=(',',':'))
    	f.close()
###############

def cross_rate_fingerprint():
    global report
    report = {}
    confusion_matrix = calculate_confusion_matrix()
    print('confusion matrix final:'+str(confusion_matrix))
    normalized_confusion_matrix = normalize_matrix(confusion_matrix)
    #number of diagonal and offdiagonal elements for NxN array  is N and (N*N-1)/2
    n_diagonal_elements = normalized_confusion_matrix.shape[0]
    n_offdiagonal_elements = float(normalized_confusion_matrix.shape[0]*normalized_confusion_matrix.shape[0]-normalized_confusion_matrix.shape[0])/2.0
    same_item_avg = mytrace(normalized_confusion_matrix)/n_diagonal_elements
    different_item_avg = (float(np.sum(normalized_confusion_matrix))-float(mytrace(normalized_confusion_matrix))) / n_offdiagonal_elements
    goodness = different_item_avg - same_item_avg
    print('same item average:'+str(same_item_avg)+' different item average:'+str(different_item_avg)+' difference:'+str(goodness))
    report['same_item_average']=same_item_avg
    report['different_item_average']=different_item_avg
    report['goodness']=goodness
    save_report(report)
    return(goodness)

def self_rate_fingerprint(fingerprint_function=fp_core.fp,weights=np.ones(fingerprint_length),distance_function=NNSearch.distance_1_k,distance_power=1.5):
    print('s.fp_func:'+str(fingerprint_function))
    print('s.weights:'+str(weights))
    print('s.distance_function:'+str(distance_function))
    print('s.distance_power:'+str(distance_power))
    global report
    report = {}
    confusion_vector = calculate_self_confusion_vector(fingerprint_function=fingerprint_function,weights=weights,distance_function=distance_function,distance_power=distance_power)

    print('confusion vector final:'+str(confusion_vector))
    n_elements = len(confusion_vector)
    same_item_avg = np.sum(confusion_vector)/n_elements
    print('unweighted same item average:'+str(same_item_avg))
    report['average_unweighted']=same_item_avg
    print('report:'+str(report))
    save_report(report)
    return(same_item_avg)



if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    weights=np.ones(fingerprint_length)/2
    self_rate_fingerprint(fingerprint_function=fp_core.fp,weights=weights,distance_function=NNSearch.distance_1_k,distance_power=1.5)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()  
    print(s.getvalue())


