#TODO - write test functions for main functions here

__author__ = 'jeremy'
from scipy import spatial
import scipy
import numpy as np
import sys
import random
import json  #needed to parse command line input as dict instead of string
import pdb
import constants
from operator import itemgetter#, attrgetter
import logging

K = constants.K # .5 is the same as Euclidean
FP_KEY = "fingerPrintVector"


def min_euclidean_distance(testPoint, Points, nMatches):
    d = len(testPoint)
    d2 = len(Points[0])
    if d != d2:
        exit('dimension of target isnt same as dimension of dB points')   #dimension mismatch
    nPoints = len(Points)
  #  print('dimMatch:'+str(d)+' dimPoints:'+str(d2)+' nPoints:'+str(nPoints)+' nMatches:'+str(nMatches))
    if nMatches > nPoints:
        print('warning:nMatches>nPoints, returning nPoints instead')
        return(min_euclidean_distance(testPoint,Points,nPoints))
    if nPoints==0:
        exit('no dB points given!')


    #make initial list of matches from first nMatches points, and sort it
  #  bestMatches=np.zeros(nMatches,np.int)    #these are the indices of the best matches
   # bestMatchValues=np.zeros(nMatches,np.double)
    tupleList=[]
    for i in range(0,nMatches):
    #    bestMatches[i]=i
        r=scipy.spatial.distance.pdist([Points[i],testPoint], metric='euclidean')[0]
 	tupleList.append( (i,r))
#    print('tuple list:'+str(tupleList))
    tupleList=sorted(tupleList,key=itemgetter(1))
#    print('sorted tuple list:'+str(tupleList))
#    bestMatches,distances=mySort(bestMatches,Points,testPoint)
 #   print('worst match:'+str(worstBestMatch))
#    for i in range(0,nMatches):
#        print(bestMatches[i],bestMatchValues[bestMatches[i]])
#    pdb.set_trace()

    worstTuple=tupleList[nMatches-1]
    worstBestMatch=worstTuple[1]
 #   print('worst:'+str(worstBestMatch))
    for i in range(nMatches,nPoints):
        r=scipy.spatial.distance.pdist([testPoint,Points[i]], metric='euclidean')[0]
 #       print(Points[i],r)
        if r<worstBestMatch:
            #insert new value in place of worst
            newTuple=(i,r)
#	    print('better tuple found:'+str(newTuple))
	    tupleList=insert(newTuple,tupleList)
	    worstTuple=tupleList[nMatches-1]
	    worstBestMatch=worstTuple[1]

#           bestMatches[nMatches-1]=i
 #           distances[nMatches-1]=r
  #          print('match switch'+str(i)+'index'+str(nMatches-1)+'value'+str(bestMatches[nMatches-1]))
  #          bestMatchValues[nMatches-1]=scipy.spatial.distance.pdist([Points[i],testPoint], metric='euclidean')[0]

            #actually its unecessary to sort every time you find a new good match, just sort at end...
            #bestMatches,worstBestMatch,worstBestMatchIndex=mySort(bestMatches,Points,testPoint)

#make list of distances
#    print('point to match:'+str(testPoint))
    bestMatches=[]
    bestMatchValues=[]
    i=0
    for tuple in tupleList:
#        r=scipy.spatial.distance.pdist([Points[bestMatches[i]],testPoint], metric='euclidean')[0]
     #   print('point '+str(i)+' index:'+str(bestMatches[i])+' point:'+str(Points[bestMatches[i]])+', distance:'+str(r))
        bestMatches.append(tuple[0])
	bestMatchValues.append(tuple[1])
	i=i+1
 #   print('best matches:'+str(bestMatches))
 #   print('best match values:'+str(bestMatchValues))
    return bestMatches, bestMatchValues


def findNNs(targetDict, entries, nMatches):
    return find_n_nearest_neighbors(targetDict, entries, nMatches, distance_1_k)

    '''
    #eliminate entries with wrong clothing class
    nEntries = len(entries)
    relevantEntries = []
#    print('target class'+str(targetDict["clothingClass"]))
    dbVectors = []
    #####eliminate entries that don't have the right clothing class
    j = 0 #this is the index for the original entries
    trackingIndex = []
#    print('target'+str(targetDict))

    for i in range(0, nEntries):
        #this is done ahead of time with the DB query
        #if entries[i]["clothingClass"]==targetDict["clothingClass"]:
        relevantEntries.append(entries[i])
        trackingIndex.append(i)
        j=j+1
        dbVectors.append(entries[i]["fingerPrintVector"])

    targetVector=targetDict["fingerPrintVector"]

    indices, distances = min_euclidean_distance(targetVector, dbVectors, nMatches)
    answerArray=[]
    answerIndices=[]
#   print('new order of indices:'+str(indices))
    for i in range(0,len(indices)):
        relevantEntries[i]['index'] = trackingIndex[indices[i]]
        answerIndices.append(trackingIndex[indices[i]])
        answerArray.append(relevantEntries[indices[i]])

    closest_matches = []
    for i in range(0, len(answerIndices)):
#        pdb.set_trace()
        match = entries[answerIndices[i]]
        match["distance"] = distances[i]
        closest_matches.append(entries[answerIndices[i]])

    return closest_matches
    #print('distances'+str(distances))
    #print('indices'+str(indices))
    '''


def find_n_nearest_neighbors(target_dict, entries, number_of_matches, distance_function):
    # list of tuples with (entry,distance). Initialize with first n distance values
    nearest_n = [(entries[i], distance_function(entries[i][FP_KEY], target_dict[FP_KEY]))
                 for i in range(0, number_of_matches)]
    nearest_n.sort(key=lambda tup: tup[1])
    # last item in the list (index -1, go python!)
    farthest_nearest = nearest_n[-1][1]

    # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
    for i in range(number_of_matches, len(entries)):
        d = distance_1_k(entries[i][FP_KEY], target_dict[FP_KEY], 1.5)
        if d < farthest_nearest:
            insert_at = number_of_matches-2
            while d < nearest_n[insert_at][1]:
                insert_at -= 1
                if insert_at == 0:
                    break
            nearest_n.insert(insert_at, (entries[i], d))
            nearest_n.pop()
            farthest_nearest = nearest_n[-1][1]
    return nearest_n


def distance_1_k(fp1, fp2, k=K):
    """This calculates distance between to arrays. When k = .5 this is the same as Euclidean."""
    if fp1 is not None and fp2 is not None:
        f12 = np.abs(np.array(fp1) - np.array(fp2))
        f12_p = np.power(f12, 1 / k)
        return np.power(np.sum(f12_p), k)
    else:
        print("null fingerprint sent to distance_1_k ")
        logging.warning("null fingerprint sent to distance_1_k ")
        return(None)


# bubble sort
'''
def mySort(indexList,Points,testPoint):
    if (len(indexList)==1):
        r=scipy.spatial.distance.pdist([Points[indexList[0]],testPoint], metric='euclidean')[0]
        worst=r
        worst_index=indexList[0]
    else:
        worst=-1
        worst_index=0 # this actually as to agree with initial assignment for the case of nMatches=1
        for i in range(0,len(indexList)-1):
            for j in range(0,len(indexList)-1-i):
                r_ij=scipy.spatial.distance.pdist([Points[indexList[j]],testPoint], metric='euclidean')[0]
                r_jk=scipy.spatial.distance.pdist([Points[indexList[j+1]],testPoint], metric='euclidean')[0]
                if r_ij>r_jk:   #swap entries
                    temp=indexList[j]
                    indexList[j]=indexList[j+1]
                    indexList[j+1]=temp
#this is actually unecessary traverse, just take liast value, its already the worst if the sort was done right
        for i in range(0,len(indexList)):
                r_ij=scipy.spatial.distance.pdist([Points[indexList[i]],testPoint], metric='euclidean')[0]
                if r_ij>worst:
                    worst=r_ij
                    worst_index=indexList[i]
#    print('sorted list')
#    print(indexList)
    return(indexList,worst,worst_index)
'''

'''
#not used
def insert(newTuple,tupleList):
    i=0
    for tuple in tupleList:
#	print('current tuple:'+str(tuple))
	if (newTuple[1]<tuple[1]):   #insert into position before
		tupleList.insert(i,newTuple)
		tupleList.pop()
#		print('list after insert:'+str(tupleList))
    		return(tupleList)
	i=i+1
#$   d=len(targetPoint)
 #   N=len(bestMatches)
  #  i=0
 #   r=scipy.spatial.distance.pdist([targetPoint,newMatch], metric='euclidean')
 #   while r< scipy.spatial.distance.pdist([bestMatches[i],newMatch], metric='euclidean'):
 #       i=i+1
 #   print(i)
#    for k in range(0,i-1):
 #       bestMatches[k,:]=bestMatches[k+1,:]
 #   bestMatches[i,:]=newMatch
    #print('bestMatches'+str(bestMatches))

      #  mindist=
    # Y : ndarray
    #Returns a condensed distance matrix Y. For each i and j (where i<j<n), the metric dist(u=X[i], v=X[j]) is computed and stored in entry ij.
'''
