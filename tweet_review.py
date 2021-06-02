# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:42:01 2021

@author: H K J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
ps=PorterStemmer()
type(ps)
dataset = pd.read_csv('reviews.tsv', delimiter='\t')
dataset=dataset[:100]
clean_reviews=[]
for i  in range(len(dataset)):
    #1 general step
    text = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    
    #2 general step
    text= text.lower()
    
    #3 general step
    text= text.split()
    
    #4 general step
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    
    text = ' '.join(text)
    
    clean_reviews.append(text)

#creating model

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
X =cv.fit_transform(clean_reviews)
X=X.toarray()
y=dataset['Liked'].values

from sklearn.linear_model import LogisticRegression
log_reg= LogisticRegression()
log_reg.fit(X,y)

y_pred= log_reg.predict(X)

from sklearn.metrics import  confusion_matrix
cm=confusion_matrix(y,y_pred)



def predict_review():
    new_review = input("Please enter the review: ")
    print(new_review)

    new_review = [new_review]
    new_review = list(new_review)
    
    X_new = cv.transform(new_review)
    X_new = X_new.toarray()
    
    new_pred = log_reg.predict(X_new)
    
    if new_pred:
        print("Positive Review")
    else:
        print("Negative Review")

predict_review()

print(cv.get_feature_names())

