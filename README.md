# Action-Recognition
The model is trained on neural networks using the mediapipe coordinates for each action. The model can recognise 7 actions. The dataset is created by capturing the
mediapipe coordinates of pose and face landmarks. Each action is trained for 30 videos and each video is of 30 frames. All the captured coordinates are saved in a 
numpy file. The model has a quite good accuracy even with this small dataset.

The second version of this model is trained by capturing the coordinates and saving them in a csv file. Compared to the previous version the results and predictions are 
quite accurate. In this rather fixing the number of frames. I have considered the complete one single cycle of each action in various poses. This version can recognise 
6 actions mainly.
