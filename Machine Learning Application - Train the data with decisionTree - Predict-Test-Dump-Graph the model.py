#!/usr/bin/env python
# coding: utf-8

# In[192]:


import pandas as pd #imports pandas library 
from sklearn.tree import DecisionTreeClassifier #imports DecisionTree algorithm
from sklearn.model_selection import train_test_split # imports the function to split the data
from sklearn.metrics import accuracy_score#imports the accuracy function to calculate the accuracy 
import joblib #this is to save the trained data and models 
from sklearn import tree #this is to export the decision tree in a graphical format 


# In[261]:


movie_data = pd.read_csv('/users/f/desktop/Predict_Movies.csv') #read the csv file and load the data. 
X = movie_data.drop(columns=['Genre']) # We need to split the data in order to create our input set from Age and Gender so we need to drop the output.
y = movie_data['Genre'] # This is to create output data. 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) #we need to split the data into (training data and testing data)
model = DecisionTreeClassifier() #Creates our model 
model.fit(X_train,y_train) #trains the data 
joblib.dump(model, 'recommend_movie.joblib')#Saves the trained model so we dont need to train it every time. 
#model = joblib.load('recommend_movie.joblib') #this is to load the saved model 
prediction = model.predict(X_test)#creates prediction results from test data 
accuracy_score(y_test, prediction)


# In[267]:


tree.export_graphviz(model, out_file='recommend_movie.dot', 
                     feature_names = ['Age', 'Gender'], 
                     class_names = sorted(y.unique()),
                     label= 'all',
                     rounded = True,
                     filled = True) #This is to get the graphic for the decisionTree model. 
#Find the created recommend_movie.dot file on your computer and open it with Visual Studio Code. 
#From the extensions section of your Visual Studio Code download (Graphviz (dot) language support for Visual Studio..)
#Right click to the opened code of (recommend_movie.dot file) and select Open Preview to the Slide 

