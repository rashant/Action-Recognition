import cv2
import mediapipe as mp
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

sequence = []
sentence = []
predictions = []

actions = ["Happy", "Sad", "Dizzy", "Thinking", "ThankYou", "Greeting", "Victory"]

threshold = 0.8
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))
model.load_weights('model.h5')


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    left_hand = np.array([[res.x, res.y, res.z] for res in
                          results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in
                           results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, left_hand, right_hand])


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 255, 255), (128, 0, 0), (0, 0, 128), (0, 128, 128),
          (128, 128, 0)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():

        # Read feed
        ret, image = cap.read()
        image = cv2.flip(image, 1)

        # Make detections
        results = holistic.process(image)
        print(results)

        # Face Landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 110), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(60, 250, 125), thickness=1, circle_radius=1))

        # Left Hand Landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(60, 22, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(60, 44, 125), thickness=1, circle_radius=1))

        # Right Hand Landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 20), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(121, 64, 250), thickness=1, circle_radius=1))

        # Pose Landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 120, 70), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(245, 70, 250), thickness=1, circle_radius=1))

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image,actions[np.argmax(res)], (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
