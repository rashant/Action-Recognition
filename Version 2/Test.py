import os
from keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
from keras.layers import *
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
r'''.\venv\Scripts\activate'''
r'''tensorboard --logdir=.'''

df = pd.read_csv('coords.csv')
x = df.drop('class', axis=1)
y = df['class']
action = ['Happy', 'Sad' ,'Greeting' ,'Victory', 'Wakanda Forever' ,'Thinking']

for i in range(len(y)):
    y[i]=action.index(y[i])
print(y[:5])
y = to_categorical(y, num_classes=6)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1234)

from keras.models import load_model
model=load_model('model_with_checkpoint.h5')
y_pred=model.predict(x_test)
for i in y_pred:
    print(action[np.argmax(i)])