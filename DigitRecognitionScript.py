#Importing numpy as np.
import numpy as np
#Importing gzip for use with .gz files.
import gzip
#Import keras for use with neural network.
import keras as kr
#Importing sklearn.preprocessing for encoding categorical variables.
import sklearn.preprocessing as pre


#Starting a neural network, building it by layers.
model = kr.models.Sequential()

#Adding a hidden layer with 1000 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))

model.add(kr.layers.Dense(units=400, activation='relu'))

#Adding a three neuron output layer.
model.add(kr.layers.Dense(units=10, activation='softmax'))

#Building the graph.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Using gzip to open the image in the data file to train the model
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

#Using gzip to open the image in the data file to train the model
with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()

train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

inputs = train_img.reshape(60000, 784)

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

model.fit(inputs, outputs, epochs=2, batch_size=100)

