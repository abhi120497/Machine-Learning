# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 00:30:07 2017

@author: Abhi
"""

#In this code, we are writing our own classifier, which will be a rudimental version of KNN
#THIS IS THE MILESTONE

#import random

from scipy.spatial import distance            #Used for calculationg euclidean distance between 2 points

def euc(a,b):
    return distance.euclidean(a,b)            #This function returns distance between 2 points where a is from training data and b is from testing data

class ScrappyKNN():
    def fit(self, X_train, y_train):          #Defining the fit function for the classifier. Self tells the program that the object calling the method is the same object to be used
        self.X_train=X_train
        self.y_train=y_train
    
    def predict(self, X_test):
        predictions = []                     #Predictions are returned in the form of array
        
        for row in X_test:
            label=self.closest(row)          #Finds the closest point to the testing data
            #label=random.choice(self.y_train) #Appending a random label to training data label
            predictions.append(label)            
        return predictions
        
    #the following function is used to find the closest point to the given data
    def closest(self, row):
        clos_dist=euc(row,self.X_train[0])
        clos_index=0
        
        for i in range(1,len(self.X_train)):
            dist=euc(row,self.X_train[i])
            if dist<clos_dist:
                clos_dist=dist
                clos_index=i
        return self.y_train[clos_index]
        

#importing a dataset
from sklearn import datasets
iris=datasets.load_iris()

X=iris.data
y=iris.target

#Direct way to split training and testing data
#Here test_size determines how much of the data is to be split i.e. here half
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.5)


#from sklearn.neighbors import KNeighborsClassifier
clf=ScrappyKNN()

clf.fit(X_train,y_train)
predictions=clf.predict(X_test)

print (predictions)

#To predict how accurate a classifier is, we can use the following command to compare with the actual
#targets of the test data
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test,predictions))