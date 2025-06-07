import mediapipe as mp

def check_posture(landmarks):
    left_ear = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value]
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]

    if abs(left_ear.x - left_shoulder.x) > 0.08:
        return True  # Bad posture
    return False
