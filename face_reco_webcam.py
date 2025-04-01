import numpy as np
import cv2 as cv

# Load the Haar cascade classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# List of people (labels)
people = ['ashan', 'chameen', 'heli', 'manodya', 'navidu', 'malith', 'bavika']

# Load the trained face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Open camera
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the camera is not working

    # Convert frame to grayscale (face detection works better in grayscale)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in face_rects:
        faces_roi = gray[y:y+h, x:x+w]  # Extract face ROI

        # Predict label and confidence
        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence:.2f}')

        # Display the label and confidence
        cv.putText(frame, str(people[label]), (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the output frame
    cv.imshow('Face Recognition', frame)

    # Press ESC to exit
    key = cv.waitKey(1)
    if key == 27:
        break

# Release resources
cap.release()
cv.destroyAllWindows()
cv.waitKey(1)
