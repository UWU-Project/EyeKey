import cv2
import numpy as np
import dlib

# To capture video from webcam.
cap = cv2.VideoCapture(0)

# face detector using dlib library
detector = dlib.get_frontal_face_detector()

# Face landmark shape predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    # Read the frame
    _, frame = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = detector(gray)

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # marking landmarks
        landmarks = predictor(gray, face)
        # for making Horizontal line
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        # for making Vertical line
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        # print Horizontal and Vertical Lines
        hor_line = cv2.line(frame, left_point, right_point, (0, 0, 255), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    # Display
    cv2.imshow("Frame", frame)

    # Stop if escape key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
