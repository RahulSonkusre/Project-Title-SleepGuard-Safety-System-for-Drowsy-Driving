import cv2
import time
import threading
import pyttsx3
import streamlit as st
import numpy as np

# --------------------- Setup Session State ---------------------
if "detection_running" not in st.session_state:
    st.session_state.detection_running = False

# --------------------- Voice Alert Function ---------------------
def speak_alert():
    def speak():
        try:
            # Create a local engine inside the thread to avoid conflicts
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 1.0)
            engine.say("Alert! You seem sleepy. Please stay focused.")
            engine.runAndWait()
        except Exception as e:
            print("Voice alert error:", e)

    # Use a daemon thread so it doesn't block the app
    t = threading.Thread(target=speak)
    t.daemon = True
    t.start()


# --------------------- Streamlit UI ---------------------
st.title("🛡️ SleepGuard - Eye Blink Detection")
st.markdown("Monitor your eye blinks to detect drowsiness while working or driving.")

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Detection"):
        st.session_state.detection_running = True
with col2:
    if st.button("⏹️ Stop Detection"):
        st.session_state.detection_running = False

frame_placeholder = st.empty()

# --------------------- Load Haar Cascades ---------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --------------------- Start Detection ---------------------
if st.session_state.detection_running:
    cap = cv2.VideoCapture(0)

    blink_count = 0
    eye_closed_frames = 0
    eye_closed_threshold = 3
    blink_alert_threshold = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Could not access the webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(roi_color, (x, y), (x + w, y + h), (255, 255, 0), 2)

            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 0:
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= eye_closed_threshold:
                    blink_count += 1
                    print("Blink detected!")
                eye_closed_frames = 0

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

        if blink_count >= blink_alert_threshold:
            speak_alert()
            blink_count = 0

        cv2.putText(frame, f"Blinks: {blink_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Exit condition
        if not st.session_state.get("detection_running", True):
            break

        time.sleep(0.05)

    cap.release()
    st.success("✅ Detection stopped.")
