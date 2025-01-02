import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os

model_path = 'face_recognition_model.h5'
class_labels_path = 'class_labels.json'

model = load_model(model_path)

with open(class_labels_path, 'r') as json_file:
    class_labels = json.load(json_file)

class_labels = {v: k for k, v in class_labels.items()}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'Enter' to capture a photo or 'q' to quit.")

detected_faces = {}
known_faces_folder = 'known_faces'

if not os.path.exists(known_faces_folder):
    os.makedirs(known_faces_folder)

def save_face_and_update_list(face, label):
    global detected_faces

    similar_labels = [key for key in detected_faces if key.startswith(label)]
    new_label = f"{label}_{len(similar_labels) + 1}"

    detected_faces[new_label] =+ 1

    face_filename = f"{new_label}.jpg"
    face_path = os.path.join(known_faces_folder, face_filename)

    cv2.imwrite(face_path, face)
    print(f"Saved face as {face_path}")
    print("Updated Detected Faces:", detected_faces)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('\r'):  # Enter key
        print("Photo captured!")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("No face detected. Marking as Absent.")
            continue

        x, y, w, h = faces[0]
        face = frame[y:y + h, x:x + w]

        resized_face = cv2.resize(face, (224, 224))
        normalized_face = resized_face / 255.0
        img_array = np.expand_dims(normalized_face, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        print(f"Confidence for detected face: {confidence:.2f}")

        if confidence >= 0.85:  # Confidence threshold
            label = class_labels[predicted_class]
            if label == 'person_1':
                print(f"Person Detected: {label} (Confidence: {confidence:.2f}) - Present")
                save_face_and_update_list(face, label)
            elif label == 'person_2':
                print(f"Person Detected: {label} (Confidence: {confidence:.2f}) - Present")
                save_face_and_update_list(face, label)
            else:
                print(f"Detected: {label} - Absent (Confidence: {confidence:.2f})")
                save_face_and_update_list(face, "Unknown")
        else:
            print("Unknown Person Detected. Marking as Absent.")
            save_face_and_update_list(face, "Unknown")

cap.release()
cv2.destroyAllWindows()

