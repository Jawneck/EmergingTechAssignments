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
model = Sequential()

model.add(Dense(512, activation='relu',kernel_initializer="normal", input_dim=784))
model.add(Dropout(0.1))
model.add(Dense(512, activation='relu',kernel_initializer="normal"))
model.add(Dropout(0.1))
model.add(Dense(units=10, activation='softmax'))

#Building the graph.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Using gzip to open the images in the data file to train the model
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

#Using gzip to open the labels in the data file to train the model
with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()
    
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()

#Reshape images and labels
train_img = np.array(list(train_img[16:])).reshape(60000, 1, 784).astype(np.uint8) / 255.0
train_lbl = np.array(list(train_lbl[ 8:])).astype(np.uint8)

test_img = np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_lbl = np.array(list(test_lbl[8:])).astype(np.uint8)

# convert class vectors to binary class matrices
train_lbl = kr.utils.to_categorical(train_lbl)

#Reshape image array
inputs = train_img.reshape(60000, 784)

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

#Training the model
model.fit(inputs, outputs, epochs=6, batch_size=128)


