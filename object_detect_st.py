import streamlit as st
import cv2 as cv
import numpy as np
import tempfile
from PIL import Image

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
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv.VideoCapture(tfile.name)

    stframe = st.empty()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Can't receive frame (stream end?). Exiting ...")
                break

            # Detect humans
            frame = detect_humans(frame)

            # Convert frame to RGB and display in Streamlit
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()
