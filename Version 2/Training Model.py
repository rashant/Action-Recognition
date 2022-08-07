import os
from keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
from keras.layers import *
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

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

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

call_back = ModelCheckpoint("model_with_checkpoint.h5", save_best_only=True, monitor='val_loss', verbose=1)
print(x_train.shape[0], x_train.shape[1])
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x, y, epochs=500,validation_split=0.2, callbacks=[tb_callback, call_back])
model.save("model.h5")
