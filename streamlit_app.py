#IMPORT STATEMENTS
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load the trained model
model = load_model('violence_detection_MobileNet_Lstm_model.h5')

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 129, 129
SEQUENCE_LENGTH = 10

# Function to process and annotate video simultaneously
def process_and_annotate_video(video_path, output_path='output_video.mp4'):

    #video capture object
    cap = cv2.VideoCapture(video_path)

    #Video Writer object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    frames = []
    frame_count = 0
    predictions = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
   
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        frames.append(resized_frame)
        
        if len(frames) == SEQUENCE_LENGTH:
            # Prepare the sequence and predict
            sequence = np.expand_dims(np.array(frames), axis=0) / 255.0
            pred = model.predict(sequence)[0][0]
            predictions.append(pred)
            
            # Annotate the frame with the prediction
            label = f"Violent ({pred:.2f})" if pred > 0.5 else f"Non-Violent ({pred:.2f})"
            color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Write the frame with annotation to the output video
            out.write(frame)
            
            # Slide window
            frames.pop(0)
        
        frame_count += 1
        # Update progress bar
        progress_bar.progress(min(frame_count / total_frames, 1.0))
    
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

        st.write("OUTPUT")
        st.video("output_video.mp4")  # Display the annotated video
        
        violent_frames = sum(p > 0.5 for p in predictions)
        non_violent_frames = len(predictions) - violent_frames
        st.write(f"Summary: **Violent Frames**: {violent_frames}, **Non-Violent Frames**: {non_violent_frames}")
    else:
        st.write("Could not process the video. Please upload a longer video or ensure it contains at least 10 frames.")
else:
    st.write("Please upload a video file to perform the prediction.")
