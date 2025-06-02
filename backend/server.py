import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import faiss
import pickle
from keras.models import load_model
import pandas as pd
from collections import deque
import time
from werkzeug.utils import secure_filename
from tempfile import NamedTemporaryFile
import traceback
from collections import deque
from flask_socketio import SocketIO, emit
from flask import request


def init_client_state():
    return {
        'sequence': [],
        'pause_buffer': [],
        'gesture_active': False,
        'pause_start_frame': None,
        'prev_keypoints': None,
        'motion_buffer': deque(maxlen=5),
        'frame_idx': 0
    }


SEQ_DIM = 225
EMBED_DIM = 128
MOTION_THRESHOLD = 0.02
PAUSE_DURATION_SEC = 1.0
SMOOTHING_WINDOW = 5
MIN_SEQ_LEN = 20
client_states = {}


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def build_embedding_model():
    inp = Input(shape=(None, SEQ_DIM))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(EMBED_DIM)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return Model(inp, x)

embedding_model = build_embedding_model()
embedding_model.load_weights("embedding_model.h5")
index = faiss.read_index("faiss_index.bin")
with open("label_encoder (1).pkl", "rb") as f:
    label_encoder = pickle.load(f)
y_enc = np.load("y_enc.npy")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

def extract_landmarks(results):
    pose = results.pose_landmarks.landmark if results.pose_landmarks else []
    lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
    rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []

    def flat_landmarks(landmarks, n_points, dims=3):
        if landmarks:
            data = np.array([[lm.x, lm.y, lm.z][:dims] for lm in landmarks])
            return data
        else:
            return np.zeros((n_points, dims))

    pose = flat_landmarks(pose, 33)
    lh = flat_landmarks(lh, 21)
    rh = flat_landmarks(rh, 21)

    all_landmarks = np.vstack([pose, lh, rh])  # shape (75, 3)

    if pose.shape[0] > 0:
        origin = pose[0]
        all_landmarks = all_landmarks - origin
        max_dist = np.max(np.linalg.norm(all_landmarks, axis=1))
        if max_dist > 1e-6:
            all_landmarks = all_landmarks / max_dist
    else:
        all_landmarks = all_landmarks  # remains zeros if no pose

    return all_landmarks.flatten()

def recognize_sign(embedding_model, index, label_encoder, y_enc, seq, top_k=5):
    emb = embedding_model.predict(np.expand_dims(seq, axis=0))  # Get embedding of sequence
    faiss.normalize_L2(emb)
    D, I = index.search(emb.astype('float32'), top_k)  # FAISS search for top_k nearest neighbors
    pred_label_indices = [y_enc[i] for i in I[0]]  # Map indices to labels
    pred_labels = label_encoder.inverse_transform(pred_label_indices)  # Inverse transform to get label names
    return pred_labels, D

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part in request'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400

    filename = secure_filename(video_file.filename)
    video_path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    video_file.save(video_path)
    try:
        # Use your existing function to recognize multiple gestures
        predictions = []
        def capture_prediction(pred_labels, D):
            if pred_labels is not None and len(pred_labels) > 0:
                predictions.append(pred_labels[0])  # top-1 label per gesture
                print(pred_labels)
                print(D)
        def recognize_with_callback(video_path):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(fps)
            frame_idx = 0
            gesture_active = False
            gesture_start_frame = None
            sequence = []
            pause_buffer = []
            pause_start_frame = None
            prev_keypoints = None
            motion_buffer = deque(maxlen=5)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                keypoints = extract_landmarks(results)
                motion = np.linalg.norm(keypoints - prev_keypoints) if prev_keypoints is not None else 0.0
                motion_buffer.append(motion)
                avg_motion = np.mean(motion_buffer)
                if avg_motion >= 0.02:
                    if pause_start_frame is not None:
                        pause_duration = (frame_idx - pause_start_frame) / fps
                        if pause_duration <= 0.9:
                            sequence.extend(pause_buffer)
                        else:
                            if len(sequence) >= 20:
                                gesture_seq = np.array(sequence)
                                pred_labels, D = recognize_sign(embedding_model, index, label_encoder, y_enc, gesture_seq)
                                capture_prediction(pred_labels,D)
                            gesture_active = False
                            sequence = []
                        pause_start_frame = None
                        pause_buffer = []
                    if not gesture_active:
                        gesture_active = True
                        gesture_start_frame = frame_idx
                    sequence.append(keypoints)
                else:
                    if gesture_active:
                        if pause_start_frame is None:
                            pause_start_frame = frame_idx
                        pause_buffer.append(keypoints)

                prev_keypoints = keypoints
                frame_idx += 1
            if gesture_active and len(sequence) >= 20:
                sequence.extend(pause_buffer)
                gesture_seq = np.array(sequence)
                pred_labels, D = recognize_sign(embedding_model, index, label_encoder, y_enc, gesture_seq)
                capture_prediction(pred_labels,D)
            cap.release()
        recognize_with_callback(video_path)
        os.remove(video_path)
        return jsonify({'prediction': ' '.join(predictions)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# @app.route('/predict_frame', methods=['POST'])
# def predict_frame():
#     global frame_idx, gesture_active, pause_start_frame, prev_keypoints
#     global motion_buffer, sequence, pause_buffer
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part in request'}), 400
#         frame_file = request.files['file']
#         frame_bytes = np.frombuffer(frame_file.read(), np.uint8)
#         frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
#         fps = 25  # Assume a fixed FPS or receive it from the client
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = holistic.process(image)
#         keypoints = extract_landmarks(results)
#         motion = np.linalg.norm(keypoints - prev_keypoints) if prev_keypoints is not None else 0.0
#         motion_buffer.append(motion)
#         avg_motion = np.mean(motion_buffer)
#         prediction = None
#         if avg_motion >= MOTION_THRESHOLD:
#             if pause_start_frame is not None:
#                 pause_duration = (frame_idx - pause_start_frame) / fps
#                 if pause_duration <= PAUSE_DURATION_SEC:
#                     sequence.extend(pause_buffer)
#                 else:
#                     if len(sequence) >= MIN_SEQ_LEN:
#                         gesture_seq = np.array(sequence)
#                         pred_labels, D = recognize_sign(embedding_model, index, label_encoder, y_enc, gesture_seq)
#                         prediction = pred_labels[0]
#                     sequence = []
#                 pause_buffer = []
#                 pause_start_frame = None
#             sequence.append(keypoints)
#             gesture_active = True
#         else:
#             if gesture_active:
#                 if pause_start_frame is None:
#                     pause_start_frame = frame_idx
#                 pause_buffer.append(keypoints)
#         prev_keypoints = keypoints
#         frame_idx += 1
#         if gesture_active and pause_start_frame is not None:
#             pause_duration = (frame_idx - pause_start_frame) / fps
#             if pause_duration > PAUSE_DURATION_SEC:
#                 if len(sequence) >= MIN_SEQ_LEN:
#                     sequence.extend(pause_buffer)
#                     gesture_seq = np.array(sequence)
#                     pred_labels, D = recognize_sign(embedding_model, index, label_encoder, y_enc, gesture_seq)
#                     prediction = pred_labels[0]
#                 sequence = []
#                 pause_buffer = []
#                 pause_start_frame = None
#                 gesture_active = False
#         return jsonify({'prediction': prediction if prediction else 'No Gesture Detected'})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({'error': str(e)}), 500

def process_frame_bytes_with_state(frame_bytes, state):
    try:
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        fps = 1  # Or get dynamically
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        keypoints = extract_landmarks(results)

        motion = np.linalg.norm(keypoints - state['prev_keypoints']) if state['prev_keypoints'] is not None else 0.0
        state['motion_buffer'].append(motion)
        avg_motion = np.mean(state['motion_buffer'])
        prediction = None
        
        if avg_motion >= MOTION_THRESHOLD:
            if state['pause_start_frame'] is not None:
                pause_duration = (state['frame_idx'] - state['pause_start_frame']) / fps
                if pause_duration <= PAUSE_DURATION_SEC:
                    state['sequence'].extend(state['pause_buffer'])
                else:
                    if len(state['sequence']) >= MIN_SEQ_LEN:
                        gesture_seq = np.array(state['sequence'])
                        pred_labels, D = recognize_sign(embedding_model, index, label_encoder, y_enc, gesture_seq)
                        prediction = pred_labels[0]
                    state['sequence'] = []
                state['pause_buffer'] = []
                state['pause_start_frame'] = None
            state['sequence'].append(keypoints)
            state['gesture_active'] = True
        else:
            if state['gesture_active']:
                if state['pause_start_frame'] is None:
                    state['pause_start_frame'] = state['frame_idx']
                state['pause_buffer'].append(keypoints)
        
        print(f"Frame idx: {state['frame_idx']}, avg_motion: {avg_motion:.4f}, pause_start: {state['pause_start_frame']}, seq_len: {len(state['sequence'])}")
        state['prev_keypoints'] = keypoints
        state['frame_idx'] += 1
        
        if state['gesture_active'] and state['pause_start_frame'] is not None:
            pause_duration = (state['frame_idx'] - state['pause_start_frame']) / fps
            if pause_duration > PAUSE_DURATION_SEC:
                if len(state['sequence']) >= MIN_SEQ_LEN:
                    state['sequence'].extend(state['pause_buffer'])
                    gesture_seq = np.array(state['sequence'])
                    pred_labels, D = recognize_sign(embedding_model, index, label_encoder, y_enc, gesture_seq)
                    prediction = pred_labels[0]
                    emit('prediction', {'prediction': prediction})
                    print(pred_labels)
                state['sequence'] = []
                state['pause_buffer'] = []
                state['pause_start_frame'] = None
                state['gesture_active'] = False
        
        return prediction if prediction else "No Gesture Detected"
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}"
@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    if sid in client_states:
        del client_states[sid]

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    frame_file = request.files['file']
    frame_bytes = frame_file.read()
    prediction = process_frame_bytes(frame_bytes)
    return jsonify({'prediction': prediction})


# WebSocket route

@socketio.on('frame')
def on_frame(frame_bytes):
    sid = request.sid  # unique client session ID
    
    if sid not in client_states:
        client_states[sid] = init_client_state()
    
    state = client_states[sid]
    print(sid)
    prediction = process_frame_bytes_with_state(frame_bytes, state)
    
    # emit('prediction', {'prediction': prediction})


# Avoid 404 on root
@app.route('/')
def index1():
    return "Flask + SocketIO server is running."

if __name__ == "__main__":
    import eventlet
    import eventlet.wsgi
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
