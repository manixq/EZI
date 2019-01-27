#!/usr/bin/env python
# coding: utf-8

# # Exercise 1 - Clustering

# This exercice uses scikit-learn library to find clusters in the collection of documents.

# In[1]:


import re
import math
import string
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


# See the code below. It does the text document preprocessing - tokenization, normalization and stemming

# In[2]:


def tokenizeAndNormalize(text):
    return [re.sub(r'\W+', '', s) for s in re.split(' |;|,|\t|\n|\.', text) if len(s) > 0]

def stemTokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(item) for item in tokens]

def preprocessing(text):
    return stemTokens(tokenizeAndNormalize(text))
        


# The class Collection parses dictionary and documents files. Field dictionary contains list of stemmed keywords, field documents list of texts and titles list of document titles. Additionnaly, field documentsCount contains the number of documents in collection.

# In[25]:


class Collection:
    def __init__(self, documentsPath, dictionaryPath):
        file = open(dictionaryPath,"r") 
        self.dictionary = sorted(set(preprocessing(file.read())))
        self.parseDocumentsFile(documentsPath)
        self.documentsCount = len(self.documents)

    def parseDocumentsFile(self, documentsPath):
        file = open(documentsPath,"r") 
        text = file.read()
        self.documents = text.split("\n\n")
        self.titles = [doc.split("\n")[0] for doc in self.documents]
    


# In[26]:


collection = Collection("documents-2.txt", "keywords-2.txt")


# $TODO$ Use $TfidfVectorizer$ from sklearn.feature_extraction.text to represent given documents in TF-IDF representation (as tokenizer pass preprocessing function, as vocabulary pass field dictionary from collection variable), set also parameter lowercase to True. You can later change this parameter to check if results change.
# To get TF-IDF representation call method fit_transform from TfidfVectorizer class.

# In[30]:


vectorizer = TfidfVectorizer(tokenizer=preprocessing, vocabulary=collection.dictionary, lowercase=True)
tfidfs = vectorizer.fit_transform(collection.documents)


# The method below prints clustering results in the form of list of documents titles. Groups are separated by blan line. It will be used later.

# In[32]:


def printResults(titles, groupLabels, k):
    results = [[] for i in range(k)]
    for docId in range(len(titles)):
        results[groupLabels[docId]].append(titles[docId])
    
    for group in results:
        for title in group:
            print(title)
        print("")


# Parameter k is the number of created clusters. The original dataset contains documenets from 9 classes (anaconda, animal planet, java island, java programming, meat puppets, perl, python, python snake, svd) but you can change it and observe the clustering results changes.

# In[48]:


k = 5


# $TODO$ Use KMeans class object (find proper parameter to pass k value) and method fit (with computed tfidfs as parameter). Printed results shows found clusters.

# In[49]:


kmeans = KMeans(n_clusters=k, random_state=0).fit(tfidfs)
printResults(collection.titles, kmeans.labels_, k)


# $TODO$ Do the same thing using Agglomerative Clustering (AgglomerativeClustering class). Method fit in this class requires dense array, so pass tfidfs.toarray() as parameter. Try different affinity and linkage parameters. Observe results changes. Try to find parameters which give the best results. 
# See documentation (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) for further information.

# In[45]:


#TODO try with different affinity (cosine, euclidean) and linkage (ward, average, complete)
agglomerative = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward').fit(tfidfs.toarray())

printResults(collection.titles, agglomerative.labels_, k)


# # Exercice 2 - Collaborative filtering

# Implement user-based collaborative filtering to find recommendations for new user.

# In[116]:


import scipy.stats
import numpy


# The code below reads file "movies.csv" with ratings in the following form (userId, itemId, rating) and processes it. As a result in the variable ratings there is a matrix with users as rows and items as colunms. In cells there are ratings.

# In[117]:


def readRatings(path):
    file=open(path, "r")
    lines = file.read().split("\n")
    return([[int(x) for x in line.split(",")] for line in lines if line != ""])

def processRatings(path):
    ratings = readRatings(path)
    maxUser = max([item[0] for item in ratings])
    maxItem = max([item[1] for item in ratings])
    ratMatrix = numpy.zeros((maxUser, maxItem))
    for rat in ratings:
        ratMatrix[rat[0]-1, rat[1]-1] = rat[2]
    return(ratMatrix)
ratings = processRatings("movies.csv")


# The method removeNotRated allows to reduce list of rated by leaving only these rated with both users (not to compute similarity based on zeros meaning that user did not rate given movie)

# In[118]:


def removeNotRated(item1, item2):
    indexes = [x for x in range(len(item1)) if item1[x] != 0 and item2[x] != 0]
    newList1 = []
    newList2 = []
    for i in indexes:
        newList1.append(item1[i])
        newList2.append(item2[i])
    return((newList1, newList2))


# $TODO$ implement similarity for pair of vectors (user ratings). Use Pearson correlation (scipy.stats.pearsonr). Handle situation if this method returns Nan (all rates for one user are equal). You can return -1 or 0 for these entries.

# In[119]:


def similarity(item1, item2):
    (item1, item2) = removeNotRated(item1, item2) #leaves only movies rated by both users
    lPearson = scipy.stats.pearsonr(item1,item2)
    if(math.isnan(lPearson[0])):
        return 0
    return lPearson[0]


# In[120]:


similarity(ratings[0],ratings[1])


# $TODO$ implement weighted average. RatingsCol parameter contains a column from ratings matrix (ratings of all users for one movie). Weights parameter is the array of similarities of users to current user (non-zero for k nearest neighbors, zeros for others).

# In[164]:


def weightedMean(ratingsCol, weights):
    lWeightedMean = 0
    lSumWeights = sum(weights)
    if(lSumWeights != 0):
        lWeightedMean = sum(r * w for r, w in zip(ratingsCol, weights)) / lSumWeights
        
    return lWeightedMean


# $TODO$ implement user-based collaborative filtering. Use the following steps:
#     * find similarities for all users fo given user (parameter userId). Remember not to take into consideration this user itself.
#     * sort similarities descending
#     * find weights vector - similarity for k nearest users, 0 for others
#     * find predicted ratings for all items, which werent already rated by this user
#         * call weightedMean method for all columns with zeros for given user and computed in step 2 weights vector)
#         * sort predicted values descending
#     * return results in the form of sorted descending list of tuples (itemId, predicted rating)

# In[173]:


k=10 #number of closest users used for recommendation
def findRecommendationsUserBased(userId, raitingsMatrix):
    lSimilarity=[]
        
    # find similarities - setting -1 should make current user last - ignoring for k < allUsers    
    safeK = min(k, raitingsMatrix.shape[0] - 1)
    for i in range(raitingsMatrix.shape[0]):
        if i != userId:
            lSimilarity.append(similarity(raitingsMatrix[userId], raitingsMatrix[i]))
        else:
            lSimilarity.append( -1)
            
    # sort similarities descending
    lSortedIndexes = sorted(range(len(lSimilarity)), key=lambda k: lSimilarity[k], reverse=True)
    lSortedIndexes[safeK:] = [0] * (len(lSortedIndexes) - safeK)    
    
    lWeightsVector = [0] * (len(lSortedIndexes))
    for lIndex in lSortedIndexes:
        if lIndex > 0:
            lWeightsVector[lIndex] = lSimilarity[lIndex]
        else:
            break;
                             
    lPredictedRatings = []
    for i in range(raitingsMatrix.shape[1]):        
        lPredictedRatings.append((i, weightedMean([row[i] for row in raitingsMatrix], lWeightsVector)))
    lPredictedRatings.sort(key=lambda tup: tup[1], reverse=True)   
    
    return lPredictedRatings


# The following code fragment prints 10 recommended movies for 10 first users. Notice that the user and movie IDs corespond the ones from input file, not the matrix indices. The matrix row/column index = user/movie ID - 1

# In[174]:


usersCount = ratings.shape[0]
for user in range(5):
    recommendations = findRecommendationsUserBased(user, ratings)
    for i in range(10):
        print("User: " + str(user + 1) + ", Item: " + str(recommendations[i][0] + 1) + ", predicted rating: " + str(round(recommendations[i][1], 2)))
    print("")


# In[ ]:





# In[ ]:




