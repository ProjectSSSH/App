import os
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

# Constants
SEQ_DIM = 225
EMBED_DIM = 128
MOTION_THRESHOLD = 0.02
PAUSE_DURATION_SEC = 1.0
SMOOTHING_WINDOW = 5
MIN_SEQ_LEN = 20
app = Flask(__name__)


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

# new_seq = np.load("/kaggle/input/outputdata/processed_npys1-20250523T102325Z-1-001/processed_npys1/B/Baby.npy")
# print("Loaded sequence shape:", new_seq.shape)
# pred_labels, distances = recognize_sign(embedding_model, index, label_encoder, y_enc, new_seq)

# Show result
# print("Top Predictions:", pred_labels)
# print("Distances:", distances)


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

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

# Function to recognize sign using the trained model and FAISS index
def recognize_sign(embedding_model, index, label_encoder, y_enc, seq, top_k=5):
    emb = embedding_model.predict(np.expand_dims(seq, axis=0))  # Get embedding of sequence
    faiss.normalize_L2(emb)
    D, I = index.search(emb.astype('float32'), top_k)  # FAISS search for top_k nearest neighbors
    pred_label_indices = [y_enc[i] for i in I[0]]  # Map indices to labels
    pred_labels = label_encoder.inverse_transform(pred_label_indices)  # Inverse transform to get label names
    return pred_labels, D


def recognize_multiple_gestures(video_path, motion_threshold=0.02, pause_duration_sec=1.0, smoothing_window=5, min_sequence_length=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    frame_idx = 0
    gesture_active = False
    gesture_start_frame = None
    sequence = []
    pause_buffer = []
    pause_start_frame = None
    prev_keypoints = None
    motion_buffer = deque(maxlen=smoothing_window)  # smoothing buffer for motion

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        keypoints = extract_landmarks(results)

        if prev_keypoints is not None:
            motion = np.linalg.norm(keypoints - prev_keypoints)
        else:
            motion = 0.0

        motion_buffer.append(motion)
        avg_motion = np.mean(motion_buffer)

        if avg_motion >= motion_threshold:
            # Motion detected: pause ended?
            if pause_start_frame is not None:
                pause_duration_frames = frame_idx - pause_start_frame
                pause_duration = pause_duration_frames / fps
        
                if pause_duration <= pause_duration_sec:
                    print(f"Pause ended at frame {frame_idx} after {pause_duration:.2f} seconds (too short, continuing gesture)")
                    # Add pause frames back to sequence
                    sequence.extend(pause_buffer)
                else:
                    gesture_end_frame = pause_start_frame
                    duration_sec = (gesture_end_frame - gesture_start_frame) / fps
                    print(f"Pause ended at frame {frame_idx} after {pause_duration:.2f} seconds (long pause, ending gesture)")
                    print(f"Gesture ended at frame {gesture_end_frame}")
                    print(f"Gesture Duration: {duration_sec:.2f} seconds")
        
                    if len(sequence) >= min_sequence_length:
                        gesture_seq = np.array(sequence)
                        pred_labels, D = recognize_sign(embedding_model, index, label_encoder, y_enc, gesture_seq)
                        print(f"Gesture Detected: {pred_labels}{D}")
        
                    gesture_active = False
                    sequence = []
        
                # Clear pause buffer and reset pause tracking BEFORE adding current frame
                pause_start_frame = None
                pause_buffer = []
        
            # Now add the current frame (motion frame) to the sequence
            if not gesture_active:
                gesture_active = True
                gesture_start_frame = frame_idx
                print(f"Gesture started at frame {gesture_start_frame}")
        
            sequence.append(keypoints)

        else:
            # Motion below threshold → pause detected
            if gesture_active:
                if pause_start_frame is None:
                    pause_start_frame = frame_idx
                    print(f"Pause started at frame {pause_start_frame}")
                pause_buffer.append(keypoints)
                pause_duration_frames = frame_idx - pause_start_frame
                pause_duration = pause_duration_frames / fps
                print(f"Pause duration: {pause_duration:.2f} seconds (frame {frame_idx})")

        prev_keypoints = keypoints
        frame_idx += 1

    # End of video
    # If gesture still active, process it (including any buffered pause frames)
    if gesture_active:
        # Add any buffered pause frames (since video ended, treat all as part of gesture)
        if pause_buffer:
            sequence.extend(pause_buffer)

        if len(sequence) >= min_sequence_length:
            gesture_seq = np.array(sequence)
            pred_labels, _ = recognize_sign(embedding_model, index, label_encoder, y_enc, gesture_seq)
            print(f"Gesture Detected (video end): {pred_labels}")
            duration = (frame_idx - gesture_start_frame) / fps
            print(f"Gesture Duration: {duration:.2f} seconds")

    cap.release()

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
                predictions.append(pred_labels[1])
                predictions.append(pred_labels[2])
                predictions.append(pred_labels[3])
                predictions.append(pred_labels[4])
                print(D)

        # Wrap the existing function to collect prediction output
        def recognize_with_callback(video_path):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

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
                        if pause_duration <= 1.0:
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

        # Clean up uploaded file
        os.remove(video_path)

        return jsonify({'prediction': ' '.join(predictions)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    frame = cv2.imread(filepath)
    os.remove(filepath)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    keypoints = extract_landmarks(results)

    if np.count_nonzero(keypoints) < 10:
        return jsonify({'label': ' '})  # Return space if no hand detected

    # Here you can choose to use a buffer to collect a short sequence before predicting,
    # but since you're predicting per frame, we'll do a best-effort prediction
    dummy_seq = np.expand_dims(keypoints, axis=0)  # shape: (1, 225)

    # LSTM expects a sequence. Pad with zero frames if needed
    seq_len = 20
    padded_seq = np.zeros((1, seq_len, SEQ_DIM))
    padded_seq[0, -1, :] = keypoints

    pred_label, _ = recognize_sign(padded_seq[0:1])  # Only use the padded input

    return jsonify({'label': pred_label})
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
