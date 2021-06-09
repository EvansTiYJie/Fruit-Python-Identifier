#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout,Flatten,Dense


# # Process Images

# In[2]:


categories =['apple','banana','mixed','orange']

def process_img(file,numlist,features,label):
    for category in categories:
        
        for i in numlist:
            try:
                img= Image.open('%s/%s_%d.jpg' % (file,category,i)).convert('RGB')
                img=img.resize((200,200))
                img_array=np.array(img)
                x=features.append(img_array)
                y=label.append(category)
                       
            except Exception as e:
                pass
        
        continue        


# In[3]:


#set up train & test data
x_train=[]
y_train=[]
x_test=[]
y_test=[]

process_img('train',range(1,77),x_train,y_train)
process_img('test',range(21,96),x_test,y_test)  


# In[4]:


#convert to numpy array
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)


# In[5]:


print(x_test)
print(y_test)


# In[6]:


print(x_train)
print(y_train)


# In[7]:


x_train=np.reshape(x_train,(x_train.shape[0],200,200,3))
x_test=np.reshape(x_test,(x_test.shape[0],200,200,3))

x_train = x_train/255
x_test=x_test/255


# In[8]:


#one_hot encoding
encoder=LabelEncoder()
train=encoder.fit_transform(y_train)
test=encoder.fit_transform(y_test)

y_train=tf.keras.utils.to_categorical(train,4)
y_test=tf.keras.utils.to_categorical(test,4)


# In[9]:


print (y_test)


# # Run CNN Model

# In[20]:


model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(200,200,3)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[22]:


model.fit(x_train,y_train,epochs=20,validation_split=0.1)


# # Evaluate Accuracy

# In[23]:


score=model.evaluate(x_test,y_test)
print('score=',score)

