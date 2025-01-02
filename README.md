Attendance System Using Face Recognition

This project is a real-time face recognition-based attendance system that identifies specific individuals and marks their presence or absence. The system uses a pre-trained deep learning model and OpenCV to detect faces and determine attendance.

Features

Real-time Face Detection: Detects faces using OpenCV's Haar Cascade Classifier.
Face Recognition: Recognizes specific individuals trained on the model.

Attendance Marking:

Marks Present if a recognized person (Person_1 or Person_2) is detected.
Marks Absent if an unknown person (Zepto or others) or no face is detected.
Confidence Threshold: Ensures that the prediction is accurate by using a confidence score.

Prerequisites

Before running the project, ensure you have the following installed:

Python 3.8 or later

pip (Python package manager)

Python Libraries

Install the required Python libraries by running:


bash

Copy code

pip install -r requirements.txt

Contents of requirements.txt:

plaintext

Copy code

tensorflow

opencv-python

numpy

Project Structure

The project is organized as follows:

plaintext

attendance_system/

│
├── dataset/

│   ├── Person_1/       # Images of Person_1 (your face)

│   ├── Person_2/       # Images of Person_2 (your friend's face)

│   └── Zepto/          # Images of others (unknown faces)

│
├── face_recognition_model.h5   # Trained model file

├── class_labels.json           # Class labels for the model

├── train.py                    # Script to train the model

├── main.py                     # Main script to run the attendance system

└── README.md                   # Project documentation

How to Use

Step 1: Prepare the Dataset

Create a folder called dataset/ in the project directory.

Add images of:

Person_1 (your face) in dataset/Person_1/.

Person_2 (your friend's face) in dataset/Person_2/.

Others (unknown faces) in dataset/Zepto/.

Each folder should contain at least 50-100 images for better accuracy.

Step 2: Train the Model

Open the train.py script.

Run the script to train the model:

bash

python train.py

The script will save the following files:

face_recognition_model.h5: The trained model.

class_labels.json: Class indices used for predictions.

Step 3: Run the Attendance System

Open the main.py script.


Run the script to start the attendance system:

bash

python main.py

Commands:

Press Enter to capture a photo and mark attendance.
Press q to exit the system.
Workflow
Face Detection:

The system uses OpenCV's Haar Cascade to detect faces in the frame.
If no face is detected, it marks the person as Absent.
Face Recognition:

The detected face is preprocessed (resized and normalized) and passed to the trained model.
The model predicts the class of the face (Person_1, Person_2, or Zepto).
If confidence is above the threshold (0.9 by default):
Person_1 or Person_2: Mark as Present.
Zepto: Mark as Absent.
Mark Attendance:

The script outputs the result in the terminal.
Example Output
Case 1: Person_1 Detected
plaintext
Copy code
Photo captured!
Confidence for detected face: 0.92
Person Detected: Person_1 (Confidence: 0.92) - Present
Case 2: Zepto Detected (Others)
plaintext
Copy code
Photo captured!
Confidence for detected face: 0.95
Detected: Zepto - Absent (Confidence: 0.95)
Case 3: No Face Detected
plaintext
Copy code
Photo captured!
No face detected. Marking as Absent.
Notes

Dataset Quality:

Use high-quality images for training to improve model performance.
Ensure the images of Person_1 and Person_2 are varied (different angles, lighting, etc.).
Confidence Threshold:

You can adjust the confidence threshold in main.py to fine-tune predictions.
Model Performance:

If the model is not distinguishing faces properly, increase the number of images in the dataset and retrain.
Future Improvements
Add Logging: Save attendance records to a file (e.g., CSV or database).
GUI: Implement a graphical interface for better user experience.
Multi-Person Detection: Allow detection of multiple faces in a frame simultaneously.
License
