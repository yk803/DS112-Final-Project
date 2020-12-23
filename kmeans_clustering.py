#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:01:54 2020

@author: Yukai Yang yy2949
"""
#%%Q0: Preparation works
# Import all the packages we need.
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
#from sklearn.impute import 

# Open the file first
df = pd.read_csv("middleSchoolData.csv");

#%%

# from kneed import KneeLocator
# from sklearn.datasets import make_blobs;
def kmeans_clustering(data):
    from sklearn.cluster import KMeans;
    from sklearn.metrics import silhouette_score;
    from sklearn.preprocessing import StandardScaler;


#features,true_labels = make_blobs(n_samples=594,centers=None); #None works as 3 here
#print(features[:5]);
#print(true_labels[:5]);

    # data_temp = np.array([df.applications,df.acceptances]).transpose();
    data_temp = data;
    scaler = StandardScaler();
#scaled_features = scaler.fit_transform(features);
    scaled_data = scaler.fit_transform(data_temp);

#print(scaled_features[:5]);

    #kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300);
    #kmeans.fit(scaled_data);

    #print(kmeans.cluster_centers_);
    silhouette_coefficients = [];
    best_labels = None;

    for k in range(2,9):
        kmeans = KMeans(n_clusters=k);
        kmeans.fit(scaled_data);
        score = silhouette_score(scaled_data,kmeans.labels_);
        silhouette_coefficients.append(score);
        label = kmeans.fit_predict(scaled_data);
        centeroids = kmeans.cluster_centers_;
        u_labels = np.unique(label);
        plt.figure();
        for i in u_labels:
            plt.scatter(data[label==i,0],data[label==i,1],label=i+1);
        
        for elem in centeroids:
            plt.scatter(elem[0]*data[:,0].std()+data[:,0].mean(), elem[1]*data[:,1].std()+data[:,1].mean(), marker="x", color='black');
        plt.legend();
        plt.show();
    
    
    plt.style.use('fivethirtyeight');
    plt.plot(range(2,9),silhouette_coefficients);
    plt.xticks(range(2,9));
    plt.xlabel("Number of Clusters");
    plt.ylabel("Silhouette Coefficient");
    plt.show();
    
    #plt.figure();
    #for i in best_labels[1]:
        #plt.scatter(data[label==i,0],data[label==i,1],label=i);
    #plt.legend();
    #plt.show();
    return kmeans
    
