import cv2
import numpy as np
import dlib
from math import hypot
# import pyglet
# import time

# To capture video from webcam.
cap = cv2.VideoCapture(0)

board = np.zeros((300, 1400), np.uint8)
board[:] = 255

# face detector using dlib library
detector = dlib.get_frontal_face_detector()
# Face landmark shape predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Keyboard settings
# Creating the Black Screen for Keyboard
keyboard = np.zeros((600, 750, 3), np.uint8)
keys_set_1 = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5",
              5: "Q", 6: "W", 7: "E", 8: "R", 9: "T",
              10: "A", 11: "S", 12: "D", 13: "F", 14: "G",
              15: "Z", 16: "X", 17: "C", 18: "V", 19: ">"}

keys_set_2 = {0: "6", 1: "7", 2: "8", 3: "9", 4: "0",
              5: "Y", 6: "U", 7: "I", 8: "O", 9: "P",
              10: "H", 11: "J", 12: "K", 13: "L", 14: "/",
              15: "B", 16: "N", 17: "M", 18: "_", 19: "<"}


def draw_letters(letter_index, text, letter_light):

    # Keys for 1st line
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 150
        y = 0
    elif letter_index == 2:
        x = 300
        y = 0
    elif letter_index == 3:
        x = 450
        y = 0
    elif letter_index == 4:
        x = 600
        y = 0

    # Keys for 2nd line
    elif letter_index == 5:
        x = 0
        y = 150
    elif letter_index == 6:
        x = 150
        y = 150
    elif letter_index == 7:
        x = 300
        y = 150
    elif letter_index == 8:
        x = 450
        y = 150
    elif letter_index == 9:
        x = 600
        y = 150

    # Keys for 3rd line
    elif letter_index == 10:
        x = 0
        y = 300
    elif letter_index == 11:
        x = 150
        y = 300
    elif letter_index == 12:
        x = 300
        y = 300
    elif letter_index == 13:
        x = 450
        y = 300
    elif letter_index == 14:
        x = 600
        y = 300
    elif letter_index == 15:
        x = 0
        y = 450
    elif letter_index == 16:
        x = 150
        y = 450
    elif letter_index == 17:
        x = 300
        y = 450
    elif letter_index == 18:
        x = 450
        y = 450
    elif letter_index == 19:
        x = 600
        y = 450

    width = 150
    height = 150
    th = 3 # thickness

    # Keyboard Text Settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 8
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    # Keyboard Key Light up color
    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (0, 255, 0), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)


def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4    # thickness lines
    cv2.line(keyboard, (int(cols/2) - int(th_lines/2), 0), (int(cols/2) - int(th_lines/2), rows), (255, 0, 255), th_lines)
    cv2.putText(keyboard, "LEFT", (55, 300), font, 6, (0, 255, 255), 5)
    cv2.putText(keyboard, "RIGHT", (50 + int(cols/2), 300), font, 6, (0, 255, 255), 5)


# Calculate the Ratio between width and the Height of the Eye
def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


font = cv2.FONT_HERSHEY_PLAIN


def get_blinking_ratio(eye_points, facial_landmarks):
    # for making Horizontal line
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    # for making Vertical line
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # print Horizontal and Vertical Lines
    # hor_line = cv2.line(frame, left_point, right_point, (0, 0, 255), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


# Draw Red Regions around the Eyes
def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye


# function to get Gaze ratio using left side white and right side white
def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    # create a black screen for eye
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])

    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]

    # gray scale the obtained eye
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    # left side threshold counting
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    # right side threshold counting
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    # to Overcome non zero error while
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


# Counters
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 7
frames_active_letter = 11


# Text and keyboard settings (Default Keyboard selection / Last Keyboard Selection)
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_frames = 0


while True:
    # Read the frame
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
    rows, cols, _ = frame.shape
    keyboard[:] = (0, 0, 0)
    frames += 1
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw a white space for loading bar
    frame[rows - 50: rows, 0: cols] = (255, 255, 255)

    if select_keyboard_menu is True:
        draw_menu()

    # Keyboard selected
    if keyboard_selected == "left":
        keys_set = keys_set_1
    else:
        keys_set = keys_set_2
    active_letter = keys_set[letter_index]

    # Face detection
    faces = detector(gray)

    for face in faces:
        # marking landmarks
        landmarks = predictor(gray, face)

        left_eye, right_eye = eyes_contour_points(landmarks)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Eyes color
        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)

        if select_keyboard_menu is True:
            # Detecting gaze to select Left or Right keybaord
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

            # Right Side looking - Gaze Ratio
            if gaze_ratio <= 0.9:
                keyboard_selected = "right"
                keyboard_selection_frames += 1
                cv2.putText(frame, "RIGHT", (50, 150), font, 3, (0, 0, 255), 4)
                # If Kept gaze on one side more than 20 frames, move to keyboard
                if keyboard_selection_frames == 20:
                    select_keyboard_menu = False

                    # Set frames count to 0 when keyboard selected
                    frames = 0
                    keyboard_selection_frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0

            # Left Side looking - Gaze Ratio
            else:
                keyboard_selected = "left"
                keyboard_selection_frames += 1
                cv2.putText(frame, "LEFT", (50, 150), font, 3, (0, 0, 255), 4)
                # If Kept gaze on one side more than 20 frames, move to keyboard
                if keyboard_selection_frames == 20:
                    select_keyboard_menu = False

                    # Set frames count to 0 when keyboard selected
                    frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0

        else:
            # Detect the blinking to select the key that is lighting up
            if blinking_ratio > 5:
                cv2.putText(frame, "BLINKING", (50, 150), font, 4, (0, 255, 0), thickness=3)
                # print(blinking_ratio)
                blinking_frames += 1
                frames -= 1

                # Show green eyes when closed
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

                # Typing letter
                if blinking_frames == frames_to_blink:
                    if active_letter != ">" and active_letter != "<" and active_letter != "_":
                        text += active_letter
                    if active_letter == "_":
                        text += " "

                    select_keyboard_menu = True

            else:
                blinking_frames = 0


# Display lightup key letters on the keyboard
    if select_keyboard_menu is False:
        if frames == frames_active_letter:
            letter_index += 1
            frames = 0
        if letter_index == 20:
            letter_index = 0
        for i in range(20):
            if i == letter_index:
                light = True
            else:
                light = False
            draw_letters(i, keys_set[i], light)

    # Show the text we're writing on the board
    cv2.putText(board, text, (80, 100), font, 4, 0, 3)

    # Blinking loading bar
    percentage_blinking = blinking_frames / frames_to_blink
    loading_x = int(cols * percentage_blinking)
    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (0, 255, 0), -1)

    # Display
    cv2.imshow("Frame", frame)
    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Board", board)

    # Stop if escape key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()