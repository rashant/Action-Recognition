"""IMPORTS"""
import os
import cv2
import mediapipe as mp
import numpy as np

"""INITIALIZING MEDIAPIPE FOR LANDMARK DETECTIONS"""
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

"""INITIATING TRAINING DATA VARIABLES"""

classes=["Happy", "Sad", "Dizzy", "Thinking", "ThankYou", "Greeting", "Victory"]
no_videos = 30
no_frames = 30
DATA_PATH = "Dataset"

cap = cv2.VideoCapture(0)

"""EXTRACT LANDMARK FEATURES"""


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


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for clas in classes:
        for vid in range(no_videos):
            for frames in range(no_frames):
                ret, image = cap.read()
                image = cv2.flip(image, 1)
                results = holistic.process(image)

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

                if frames == 0:
                    cv2.putText(image, "STARTING COLLECTION FOR NEW VIDEO FOR {}".format(clas), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 5, cv2.LINE_AA)
                    cv2.putText(image, "Collecting frames for {} Video Number{} Frame Number {}".format(clas, vid + 1,
                                                                                                        frames + 1),
                                (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.waitKey(5000)

                else:
                    cv2.putText(image, "Collecting frames for {} Video Number{} Frame Number {}".format(clas, vid + 1,
                                                                                                        frames + 1),
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                keypoints = extract_keypoints(results)
                numpy_file_path = os.path.join(DATA_PATH, clas, str(vid), str(frames))
                try:
                    np.save(numpy_file_path, keypoints)
                except FileNotFoundError as e:
                    os.mkdir(os.path.join(DATA_PATH, clas, str(vid)))
                    np.save(numpy_file_path, keypoints)

                cv2.imshow("Pose Detection", image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

