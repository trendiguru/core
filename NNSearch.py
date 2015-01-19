__author__ = 'jeremy'
from scipy import spatial
import scipy
import numpy as np
import sys
import random
import json  #needed to parse command line input as dict instead of string
import pdb
from operator import itemgetter#, attrgetter

#>>> sorted(student_tuples, key=itemgetter(2))
#[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
#>>> sorted(student_objects, key=attrgetter('age'))
#[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]

def MinEuclideanDistance(testPoint, Points, nMatches):
    d = len(testPoint)
    d2 = len(Points[0])
    if d != d2:
        exit('dimension of target isnt same as dimension of dB points')   #dimension mismatch
    nPoints = len(Points)
  #  print('dimMatch:'+str(d)+' dimPoints:'+str(d2)+' nPoints:'+str(nPoints)+' nMatches:'+str(nMatches))
    if nMatches > nPoints:
        print('warning:nMatches>nPoints, returning nPoints instead')
        return(MinEuclideanDistance(testPoint,Points,nPoints))
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


#add use of bestmatchvalues to avoid repeatedly calculating euclidean distance

# bubble sort
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

def findNNs(targetDict, entries, nMatches):
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

    indices, distances = MinEuclideanDistance(targetVector, dbVectors, nMatches)
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




# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################
# #        NNSearch
# ###############################################################################################################################
# ###############################################################################################################################
# #
# # command line usage: python NNSearch.py '{"fingerPrintVector":[list of real],"clothingClass":[list of int]}' nMatches
# #
# #the clothingClass is a vector , currently length 1 but getting longer as our dict power grows and we become more excited about clothing recognition
# #targetDict = {"clothingClass":list,"fingerPrintVector":list}   where the lists are arrays - integer for clothing class, real for fingerprint
#
# vectorLength=2  #length of the fingerprint vector - this only needs to be specified during testing when we have to fake
#                 #both the target and the db entries
#
# #get target from command line
# #usage: python NNSearch.py '{"fingerPrintVector":[0.1,0.2],"clothingClass":[1]}' 2
#
# nArgs=len(sys.argv)
# if nArgs>1:
#     targetDict = json.loads(sys.argv[1])
#     if nArgs>2:
#         nMatches=int(sys.argv[2])
#     else:
#         nMatches=1   #if command line used for dict but not nMatches assume one (best) match is desired
# #    print('target:'+str(targetDict)+' nMatches:'+str(nMatches))
# #unless there's no command line argument in which case make up a random target and make up nMatches
# else:
#     vector=[]
#     nMatches=3  #look for top 3 matches
#     for j in range(0,vectorLength):
#         r=random.random()
#         vector.append(r)
#     clothingClass=random.randint(0,3)
#     targetDict = {"clothingClass":[clothingClass],"fingerPrintVector":vector}
#     #print(targetDict)
#  #print(len(targetDict["fingerPrintVector"]))
#
# # and an array of  dicts of potential matches from comes from the dB
# #this is a fake in lieu of real dictionary data
# #Generate fake dictionary array
# nEntries=20  #number of dictionary entries to generate
# entries=[]
# for i in range(0,nEntries):
#     vector=[]
#     for j in range(0,vectorLength):
#         r=random.random()
#         vector.append(r)
#     clothingClass=[random.randint(0,3)]
#     entryDict={"clothingClass":clothingClass,"fingerPrintVector":vector}
#     entries.append(entryDict)
# #print('initial entries:'+str(entries))
# #print('target:'+str(targetDict))
# #print('nMatches:'+str(nMatches))
# answers,answerIndices=findNNs(targetDict,entries,nMatches)
# #print('answer:'+str(answers))
# #print('answer indices:'+str(answerIndices))
# print(answerIndices)


