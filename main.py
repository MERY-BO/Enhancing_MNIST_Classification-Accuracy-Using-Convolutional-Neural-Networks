import tensorflow as tf
from tensorflow import keras
import numpy as np


from keras.datasets import fashion_mnist


(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
x_train,x_test = x_train / 255,x_test / 255

class myCallbacks(tf.keras.callbacks.Callback):
  def on_epoch(self, epoch, logs = {}):
    if logs.get('loss') < 0.04:
      print('model gonna stop less than 0.4! ')
      self.model.stop_training = True

callback = myCallbacks()
model = keras.Sequential([
  keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape = (28,28,1)),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Conv2D(64,(3,3),activation = 'relu'),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Flatten(),
  keras.layers.Dense(128,activation = 'relu'),
  keras.layers.Dense(10,activation = 'softmax')
])


model.summary()
model.compile(optimizer ='sgd', loss = 'mean_squared_error',metrics = ['accuracy'])

model.fit(x_train,y_train,epochs = 3, callbacks = [callback])

