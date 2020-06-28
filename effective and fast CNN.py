from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import  tensorflow as tf
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layer = '0'
layer_size = '64'
conv_layer = '3'


NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
print(NAME)



# initialize the model
classes=122
model = Sequential()
# first set of convolutional layer.
model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# second set convolutional layer.
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#we will dropout 20% of the neurons to improve generalization.
model.add(Dropout(0.2))
# Flatten layer
model.add(Flatten())
# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
# Output layer
model.add(Dense(classes, activation='softmax'))





model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))

model.fit(X, y, batch_size=32, epochs=30, callbacks=[tensorboard])

model.save('64x3-CNN.model')
