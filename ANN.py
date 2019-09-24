#!/usr/bin/env python
# coding: utf-8

# In[52]:


#Artificial neural network
#step - 1 Data Pre-processing
#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[53]:


#Importing the dataset
df = pd.read_csv("C:/Users/kanch/OneDrive/Desktop/project/CC project/Churn_Modelling.csv")


# In[54]:


df.head()


# In[55]:


x = df.iloc[: , 3:13]
y = df.iloc[ : ,13]


# In[56]:


#Create dummy variables
geography = pd.get_dummies(x['Geography'], drop_first = True)
gender = pd.get_dummies(x['Gender'], drop_first = True)


# In[ ]:


# Concatinate the dataframes
x = pd.concat([x,geography, gender],axis =1)


# In[59]:


#Drop unnecessary columns
x = x.drop(['Geography','Gender'], axis =1)


# In[66]:


#Split the data into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2 , random_state = 0)


# In[70]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[77]:


#Now lets make the Artificial Neural Network(ANN)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# In[78]:


#Initialising ANN
classifier = Sequential()


# In[80]:


#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu',input_dim = 11))


# In[82]:


#Adding the Second hidden layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform', activation = 'relu'))


# In[83]:


#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))


# In[86]:


#Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'] )


# In[87]:


#Fitting the ANN to training set
model_history = classifier.fit(x_train,y_train, validation_split = 0.33, batch_size = 10, nb_epoch =100)


# In[88]:


#List all data in history
print(model_history.history.keys())


# In[92]:


#Summarize history for Accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[93]:


#summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()


# In[94]:


#Part - 3 Making the predictions and evaluating the model


# In[95]:


#predicting the test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)


# In[99]:


#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[101]:


#Calculate the accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)


# In[102]:


score


# In[ ]:




