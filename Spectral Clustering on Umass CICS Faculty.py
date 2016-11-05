# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:42:54 2016

@author: HOME
"""

import scipy
import numpy as np
from sklearn.cluster import KMeans


m = scipy.io.loadmat("D:\Learning\Umass\ML 689\Individual mini projects\graphData.mat")
Adj=m['data_struct'][0][0]
## Number of clusters to divide the data set into
clusters=3
# code to check if it is symmetrix :- (Adj.transpose() == Adj).all()
A=np.zeros((40,40))
prof=m['data_struct'][0][1]

## Forming the afffinity matrix using distances and setting diagonal elements to 0
for i in range(0,Adj.shape[0]):
    S_i=Adj[i]
    ro=0.5
    for j in range(0,Adj.shape[0]):
        S_j=Adj[j]
        A[i][j]=np.exp((-0.5)*(np.sum(np.square((S_i-S_j)))/(ro**2)))
        
for i in range(0,A.shape[0]):
    A[i][i]=0

rowsum = Adj.sum(axis=1)
# Degree matrix formed by setting diagonal elements to the row sums of Affinity matrix
D=np.diag(rowsum)
D_inv=np.linalg.inv(D)
## Calculating the Laplacian
L= (np.sqrt(D_inv)).dot(Adj.dot(np.sqrt(D_inv)))
eigenValues,eigenVectors = np.linalg.eig(L)
## Sorting eigen vectors to find the k largest ones and form matrix X by appending them as columns 
index = eigenValues.argsort()[-clusters:][::-1]
eigenValues = eigenValues[index]
eigenVectors = eigenVectors[:,index]
#X=np.zeros((len(Adj),clusters))
#for k in range(0,clusters):
#    X[:,k]=eigenVectors[:,k]

# Normalizing to get unit row lengths
Y=np.zeros((40,clusters))
for k in range(0,eigenVectors.shape[0]):
    for l in range(clusters):
        Y[k][l]= eigenVectors[k][l]/np.sqrt(np.sum(np.square(eigenVectors[k][:])))
## Using K means algorithm to cluster Y matrix       
kmeans = KMeans(n_clusters=clusters, init='k-means++', n_init =20 ,max_iter=1000, tol=0.001, precompute_distances='auto',random_state=0).fit(Y)
cluster_array=kmeans.labels_
## Assigning professors to the respective clusters as per the algorithm
final_labels=np.unique(cluster_array)
for l in range(0,cluster_array.shape[0]):
    for label in range(0,final_labels.shape[0]):
        if cluster_array[l]==final_labels[label]:
            print "Professor "+ str(prof[l][0]) + " belongs to cluster " + str(label)
        

