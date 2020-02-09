# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:09:11 2020

@author: niles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#cleaning data
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('English'))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X_data = cv.fit_transform(corpus)
X_data=X_data.toarray()
X=X_data.copy()
y=dataset.iloc[:,1].values


#classification algorithm as we have 2 types of o/p

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_data,y,test_size=0.2,random_state=29)


#naive bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

y_pred_rf = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm_rf=confusion_matrix(y_test,y_pred_rf)




