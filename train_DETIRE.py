
# coding: utf-8

# In[ ]:


import keras
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Conv2D, MaxPooling2D, Embedding,BatchNormalization
from keras.layers import Bidirectional,LSTM
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

os.chdir('your_dir')#set your own working direction
K.set_image_dim_ordering('tf')

# two set of trainable parameters for BiLSTM-path and CNN-path, respectively
class Hadamard1(Layer):

    def __init__(self, **kwargs):
        super(Hadamard1, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True)
        super(Hadamard1, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        
        return tf.multiply(self.kernel, x) 

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return input_shape    
    
class Hadamard2(Layer):

    def __init__(self, **kwargs):
        super(Hadamard2, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True)
        super(Hadamard2, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        
        return tf.multiply(1-self.kernel, x)

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return input_shape    


#hyper-parameters of the sequences
time_step=498
embedding_size=30
#hyper-parameters of the CNN-path
filter_length_1 = 7    
filter_length_2 = 5
filter_length_3 = 3
nb_filter_1 = 16       
nb_filter_2 = 32
nb_filter_3 = 64
pool_length = 4      
#hyper-parameters of the BiLSTM-path
lstm_output_size_1 = 150   
lstm_output_size_2 = 70
#hyper-parameters of the training stragety
batch_size = 200  
nb_epoch = 20  

#creating labels
a=[]
for i in range(50000): #number of viruses in the training dataset
    a.append('1')
for i in range(50000): #number of non-viruses in the training dataset
    a.append('0')
b=[]
for i in range(10000): #number of viruses in the testing dataset
    b.append('1')
for i in range(10000): #number of non-viruses in the testing dataset
    b.append('0')


X_train=np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=0)
y_train = keras.utils.to_categorical(a,num_classes=2)
X_test = np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=0)
y_test = keras.utils.to_categorical(b,num_classes=2)

#embedding
embedding = np.loadtxt(open("embedding2.csv","rb"),delimiter=",",skiprows=0) #import the embedding matrix
arr1 = np.array(X_train)
m=np.zeros(shape=(arr1.shape[0],arr1.shape[1],embedding_size))
for i in range(0,arr1.shape[0]):
    for j in range(0,arr1.shape[1]):
        k=int(arr1[i,j])
        m[i,j,:]=embedding[k]
arr2 = np.array(X_test)
n=np.zeros(shape=(arr2.shape[0],arr2.shape[1],embedding_size))
for i in range(0,arr2.shape[0]):
    for j in range(0,arr2.shape[1]):
        l=int(arr2[i,j])
        n[i,j,:]=embedding[l]


X_train_rnn=m
X_train_cnn=m.reshape(len(X_train),time_step,embedding_size,1)
X_test_rnn=n
X_test_cnn=n.reshape(len(X_test),time_step,embedding_size,1)


#models of the BiLSTM-path
s1rnn = Sequential()
s1rnn.add(Bidirectional(LSTM(lstm_output_size_2), batch_input_shape=(None,time_step, embedding_size),merge_mode='concat'))
s1rnn.add(Hadamard1())
s1rnn.summary()


#models of the CNN-path
s2cnn = Sequential()
s2cnn.add(Conv2D(nb_filter_1,(filter_length_1,embedding_size),
                        border_mode='valid',
                        input_shape=(time_step,embedding_size,1)))
s2cnn.add(MaxPooling2D(pool_size=(pool_length,1)))
s2cnn.add(BatchNormalization())
s2cnn.add(Conv2D(nb_filter_2,(filter_length_2,1),
                        border_mode='valid'))
s2cnn.add(MaxPooling2D(pool_size=(pool_length,1)))
s2cnn.add(BatchNormalization())
s2cnn.add(Conv2D(nb_filter_3,(filter_length_3,1),
                        border_mode='valid'))
s2cnn.add(MaxPooling2D(pool_size=(pool_length,1)))
s2cnn.add(BatchNormalization())
s2cnn.add(Flatten())
s2cnn.add(Hadamard2())
s2cnn.summary()
#the merged model        
model = Sequential()
model.add(Merge([s1rnn,s2cnn],mode='concat'))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))  
model.summary()   
adam=Adam(lr=0.03,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
history = model.fit([X_train_rnn,X_train_cnn], y_train, batch_size=batch_size, epochs=nb_epoch,validation_data=([X_test_rnn,X_test_cnn], y_test))
model.save('DETIRE_model.h5')

