# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# import mediapipe as mp
# import torch
# import torch.nn as nn
# import os
# from werkzeug.utils import secure_filename
# from tempfile import NamedTemporaryFile

# # ====== Flask Setup ======
# app = Flask(__name__)

# # ====== Model Setup ======
# class GestureLSTM(nn.Module):
#     def __init__(self, input_size=225, hidden_size=128, num_classes=84):
#         super(GestureLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, 64)
#         self.fc2 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = out[:, -1, :]
#         out = torch.relu(self.fc1(out))
#         return self.fc2(out)

# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = GestureLSTM(input_size=225, hidden_size=128, num_classes=1888)
# label_classes = np.load("label_classes.npy", allow_pickle=True)  # Make sure this also has 1888 classes

# model.load_state_dict(torch.load("model.pth", map_location=device))
# model.to(device)
# model.eval()


# # ====== MediaPipe Setup ======
# mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic(static_image_mode=False)

# def extract_landmarks_from_video(video_path, max_frames=100):
#     cap = cv2.VideoCapture(video_path)
#     landmarks_sequence = []

#     while len(landmarks_sequence) < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = holistic.process(frame_rgb)

#         frame_landmarks = []

#         # Extract pose landmarks
#         if results.pose_landmarks:
#             for lm in results.pose_landmarks.landmark:
#                 frame_landmarks.extend([lm.x, lm.y, lm.z])
#         else:
#             frame_landmarks.extend([0]*33*3)

#         # Extract left hand
#         if results.left_hand_landmarks:
#             for lm in results.left_hand_landmarks.landmark:
#                 frame_landmarks.extend([lm.x, lm.y, lm.z])
#         else:
#             frame_landmarks.extend([0]*21*3)

#         # Extract right hand
#         if results.right_hand_landmarks:
#             for lm in results.right_hand_landmarks.landmark:
#                 frame_landmarks.extend([lm.x, lm.y, lm.z])
#         else:
#             frame_landmarks.extend([0]*21*3)

#         landmarks_sequence.append(frame_landmarks)

#     cap.release()

#     # Pad if needed
#     while len(landmarks_sequence) < max_frames:
#         landmarks_sequence.append([0]*225)

#     return np.array(landmarks_sequence[:max_frames])

# # ====== Inference Route ======
# @app.route("/predict_video", methods=["POST"])
# def predict_video():
#     if "video" not in request.files:
#         return jsonify({"error": "No video uploaded"}), 400

#     file = request.files["video"]
#     temp_video = NamedTemporaryFile(delete=False, suffix=".mp4")
#     file.save(temp_video.name)

#     try:
#         sequence = extract_landmarks_from_video(temp_video.name)
#         input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = model(input_tensor)
#             pred_index = torch.argmax(output, dim=1).item()
#             pred_label = label_classes[pred_index]
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         os.remove(temp_video.name)

#     return jsonify({"prediction": pred_label})
# @app.route("/predict_frame", methods=["POST"])
# def predict_frame():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files["image"]
#     npimg = np.frombuffer(file.read(), np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     results = holistic.process(img_rgb)

#     frame_landmarks = []

#     # Extract pose landmarks
#     if results.pose_landmarks:
#         for lm in results.pose_landmarks.landmark:
#             frame_landmarks.extend([lm.x, lm.y, lm.z])
#     else:
#         frame_landmarks.extend([0]*33*3)

#     # Extract left hand
#     if results.left_hand_landmarks:
#         for lm in results.left_hand_landmarks.landmark:
#             frame_landmarks.extend([lm.x, lm.y, lm.z])
#     else:
#         frame_landmarks.extend([0]*21*3)

#     # Extract right hand
#     if results.right_hand_landmarks:
#         for lm in results.right_hand_landmarks.landmark:
#             frame_landmarks.extend([lm.x, lm.y, lm.z])
#     else:
#         frame_landmarks.extend([0]*21*3)

#     if len(frame_landmarks) != 225:
#         return jsonify({"error": "Failed to extract full set of landmarks"}), 500

#     # Pad to 100 frames with zeros
#     sequence = [frame_landmarks] + [[0]*225]*99
#     sequence = np.array(sequence)

#     input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(input_tensor)
#         pred_index = torch.argmax(output, dim=1).item()
#         pred_label = label_classes[pred_index]

#     return jsonify({"prediction": pred_label})

# # ====== Run App ======

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import mediapipe as mp

# ====================
# Load model and utils
# ====================

class GestureLSTM(nn.Module):
    def __init__(self, input_size=225, hidden_size=128, num_classes=10):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        return self.fc2(out)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = GestureLSTM(input_size=225, hidden_size=128, num_classes=1888)  # Adjust num_classes
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Load label classes
label_classes = np.load("label_classes.npy",allow_pickle=True)

# ==============
# Mediapipe setup
# ==============

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False)

def extract_landmarks(results):
    def flat_landmarks(landmarks, n_points, dims=3):
        if landmarks:
            data = np.array([[lm.x, lm.y, lm.z][:dims] for lm in landmarks])
            return data.flatten()
        else:
            return np.zeros(n_points * dims)
    
    pose = flat_landmarks(results.pose_landmarks.landmark if results.pose_landmarks else [], 33)
    lh = flat_landmarks(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [], 21)
    rh = flat_landmarks(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [], 21)

    return np.concatenate([pose, lh, rh])  # 33*3 + 21*3 + 21*3 = 225

def process_video(video_path, maxlen=100):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)

    cap.release()
    sequence = np.array(sequence)

    if sequence.shape[0] < maxlen:
        pad = np.zeros((maxlen - sequence.shape[0], sequence.shape[1]))
        sequence = np.vstack((sequence, pad))
    else:
        sequence = sequence[:maxlen]

    return sequence

# =============
# Flask Backend
# =============

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict_video", methods=["POST"])
def predict():
    print("Request received")
    print("Request files:", request.files)
    
    if "video" not in request.files:
        return jsonify({"error": "No video file found"}), 400
    if "video" not in request.files:
        return jsonify({"error": "No video file found"}), 400

    video = request.files["video"]
    filepath = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(filepath)

    try:
        sequence = process_video(filepath)
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = label_classes[pred]
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
