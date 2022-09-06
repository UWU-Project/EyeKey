import cv2
import numpy as np
import dlib
from math import hypot

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# face detector using dlib library
detector = dlib.get_frontal_face_detector()
# Face landmark shape predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_ITALIC


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_blinking_ratio(eye_points, facial_landmarks):
    # for making Horizontal line
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    # for making Vertical line
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[5]))

    # print Horizontal and Vertical Lines
    # hor_line = cv2.line(frame, left_point, right_point, (0, 0, 255), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

    ratio = hor_line_length / ver_line_length
    return ratio


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

        # detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5:
            cv2.putText(frame, "EyeBlink", (50, 150), font, 3, (255, 0, 0))

        # print(blinking_ratio)

        # Gaze detection
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        # print(left_eye_region)

        # create a black screen for eye
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(frame, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])

        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = left_eye[min_y: max_y, min_x: max_x]

        # gray scale the obtained eye
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape

        left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0: height, int(width/2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        gaze_ratio = left_side_white/right_side_white

        cv2.putText(frame, str(gaze_ratio), (50, 150), font, 2, (0, 0, 255), 3)
        # cv2.putText(frame, str(left_side_white), (50, 100), font, 2, (0, 0, 255), 3)
        # cv2.putText(frame, str(right_side_white), (50, 150), font, 2, (0, 0, 255), 3)

        # resize the eye
        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        eye = cv2.resize(gray_eye, None, fx=5, fy=5)

        # display the eye in new window
        cv2.imshow("Eye", eye)

        # display eye in grayscale in new window
        cv2.imshow("Threshold", threshold_eye)

        cv2.imshow("left", left_side_threshold)
        cv2.imshow("right", right_side_threshold)

        # display created black screen
        # cv2.imshow("Left Eye", left_eye)

    # Display
    cv2.imshow("Frame", frame)

    # Stop if escape key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()

# DONE P3
