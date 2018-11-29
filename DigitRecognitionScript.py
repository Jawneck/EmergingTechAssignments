#Importing numpy as np.
import numpy as np
#Importing gzip for use with .gz files.
import gzip
#Import keras for use with neural network.
import keras as kr
#Importing sklearn.preprocessing for encoding categorical variables.
import sklearn.preprocessing as pre
from keras.models import Sequential
from keras.layers import Dense, Dropout

#Starting a neural network, building it by layers.
model = kr.models.Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))

#Building the graph.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Using gzip to open the images in the data file to train the model
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

#Using gzip to open the labels in the data file to train the model
with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()

#Reshape images and labels
train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

#Reshape image array
inputs = train_img.reshape(60000, 784)

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

#Training the model
model.fit(inputs, outputs, epochs=20, batch_size=100)



