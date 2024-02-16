import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import tempfile

# Load the cascades
human_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')

def detect_humans(frame):
    """Detect humans in a video frame."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in humans:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

st.title("Human Detection in Video")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    cap = cv.VideoCapture(tfile.name)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    out = cv.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect humans
            processed_frame = detect_humans(frame)

            # Write the frame into the file 'output.mp4'
            out.write(processed_frame)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()
        out.release()
        
    # Display download link for the processed video
    with open('output.mp4', 'rb') as f:
        st.download_button('Download Processed Video', f, file_name='processed_video.mp4')
