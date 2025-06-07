from flask import Flask, render_template, Response, jsonify
import os
import cv2
import mediapipe as mp
from posture_utils import check_posture
import time

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask with custom template and static folders
app = Flask(__name__,
           template_folder=os.path.join(parent_dir, 'templates'),
           static_folder=os.path.join(parent_dir, 'static'))

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

last_alert_time = 0
alert_cooldown = 3  # seconds

# Global flag for posture status
bad_posture_flag = False

def gen_frames():
    global last_alert_time, bad_posture_flag

    while True:
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if check_posture(landmarks):
                bad_posture_flag = True
                if time.time() - last_alert_time > alert_cooldown:
                    last_alert_time = time.time()

                # Show red alert box
                cv2.putText(image, "⚠️ Bad Posture Detected!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.rectangle(image, (40, 80), (600, 140), (0, 0, 255), cv2.FILLED)
                cv2.putText(image, "Please Sit Straight!", (60, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                bad_posture_flag = False
                cv2.putText(image, "✅ Good Posture", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/posture_status')
def posture_status():
    return jsonify({"bad_posture": bad_posture_flag})

if __name__ == "__main__":
    app.run(debug=True)
