import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import requests

# Model URL
model_url = "https://raw.githubusercontent.com/Jesly-Joji/Violence-Detection/main/violence_detection_mobilenet_lstm_model.h5"

# Download and save the model
response = requests.get(model_url)
with open("model.h5", "wb") as f:
    f.write(response.content)

# Load the model
model = load_model("model.h5")

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 129, 129
SEQUENCE_LENGTH = 10

# Function to process and annotate video
def process_and_annotate_video(video_path, output_path='output_video.mp4'):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file.")
        return []
    
 
    fourcc = cv.VideoWriter_fourcc(*’avc1')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    frames = []
    predictions = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frames.append(resized_frame)
        
        # Process frames only when we have enough to form a sequence
        if len(frames) == SEQUENCE_LENGTH:
            sequence = np.expand_dims(np.array(frames), axis=0) / 255.0
            pred = model.predict(sequence)[0][0]
            predictions.append(pred)
            label = f"Violent ({pred:.2f})" if pred > 0.5 else f"Non-Violent ({pred:.2f})"
            color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            out.write(frame)
            frames.pop(0)  # Ensure this only happens if frames is not empty
        
        progress_bar.progress(min(cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames, 1.0))
    
    cap.release()
    out.release()

    return predictions

# Streamlit app
st.title("Violence Detection in Videos")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the video and annotate it
    predictions = process_and_annotate_video(video_path)

    if predictions:
        st.write("Input File")
        st.video(uploaded_file)  # Display the uploaded video

        st.write("Output File")
        st.video("output_video.mp4")  # Display the annotated video

        violent_frames = sum(p > 0.5 for p in predictions)
        non_violent_frames = len(predictions) - violent_frames
        st.write(f"Summary: **Violent Frames**: {violent_frames}, **Non-Violent Frames**: {non_violent_frames}")
    else:
        st.write("Could not process the video. Please ensure it contains at least 10 frames.")
else:
    st.write("Please upload a video file to perform the prediction.")
