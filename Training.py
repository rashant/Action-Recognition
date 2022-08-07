from keras.utils import to_categorical
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, EarlyStopping

r'''.\venv\Scripts\activate'''
r'''tensorboard --logdir=.'''

actions = ["Happy", "Sad", "Dizzy", "Thinking", "ThankYou", "Greeting", "Victory"]
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
no_videos = 30
no_frames = 30
DATA_PATH = "Dataset"

for action in actions:
    for sequence in range(no_videos):
        window = []
        for frame_num in range(no_frames):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

print(np.array(sequences, dtype="object").shape)

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)
es = EarlyStopping(monitor='loss', mode='auto', baseline=0.1, restore_best_weights=True,patience=50)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(x.shape[1], x.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x, y, epochs=500, callbacks=tb_callback)
model.save("model.h5")
