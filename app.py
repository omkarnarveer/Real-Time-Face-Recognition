import cv2
import face_recognition
import pickle
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the trained model
with open('model/model.pkl', 'rb') as f:
    model_data = pickle.load(f)

train_images = model_data['train_images']
train_labels = model_data['train_labels']

FACE_DISTANCE_THRESHOLD = 0.6  # Classification threshold

def recognize_faces(frame):
    """Detect and recognize faces in the given frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(train_images, face_encoding)
        min_distance = min(distances)

        if min_distance <= FACE_DISTANCE_THRESHOLD:
            match_index = np.argmin(distances)
            name = train_labels[match_index]
        else:
            name = "Unknown"

        # Draw a rectangle and label around the detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def generate_frames():
    """Captures video frames, detects and recognizes faces."""
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = recognize_faces(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
