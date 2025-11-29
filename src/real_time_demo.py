import warnings
import os

# 1. Suppress the library warnings to keep console clean
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ### NEW: Suppress MediaPipe/TensorFlow C++ logs ###
# 0 = all, 1 = info, 2 = warnings, 3 = errors
os.environ['GLOG_minloglevel'] = '2' 
# ###################################################

import cv2
import mediapipe as mp
import numpy as np
import joblib

# --- Configuration ---
MODEL_PATH = "../models/svm_winner.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    exit()

print("Loading model...")
svm_model = joblib.load(MODEL_PATH)
print("Model loaded.")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter("output.mp4", fourcc, 30, (frame_width, frame_height))

print("Starting video stream. Press 'q' to exit.")

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = hands.process(rgb_frame)

        rgb_frame.flags.writeable = True
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

                # Normalize (Wrist at 0,0)
                wrist_x, wrist_y, wrist_z = landmarks[0]
                landmarks[:, 0] -= wrist_x 
                landmarks[:, 1] -= wrist_y 

                # Scale (based on middle finger)
                mid_finger_x, mid_finger_y, _ = landmarks[12] 
                scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
                if scale_factor > 0:
                    landmarks[:, 0] /= scale_factor 
                    landmarks[:, 1] /= scale_factor 

                # Flatten
                features = landmarks.flatten().reshape(1, -1)

                try:
                    # Predict
                    prediction = svm_model.predict(features)[0]

                    cv2.putText(frame, f'Prediction: {prediction}', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                except Exception as e:
                    pass # Ignore prediction errors if hand is momentarily weird

                # Draw
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        out.write(frame)
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()