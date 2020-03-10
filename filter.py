import dlib;
import numpy as np;
import cv2;
from math import hypot;


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

cap = cv2.VideoCapture(0)
nose_image = cv2.imread("clown_nose.png")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

nose_pig = nose_image;
nose_area = None

while True:
    _, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        (x,y,w,h) = rect_to_bb(face)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

        landmarks = predictor(gray, face)

        top_nose = (int(landmarks.part(29).x), int(landmarks.part(29).y))
        center_nose = (int(landmarks.part(30).x), int(landmarks.part(30).y))
        left_nose = (int(landmarks.part(31).x), int(landmarks.part(31).y))
        right_nose = (int(landmarks.part(35).x), int(landmarks.part(35).y))

        nose_width = int(hypot(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) * 2)
        nose_height = int(nose_width * 0.879)

        # nose position
        top_left = int(center_nose[0] - nose_width / 2), int(center_nose[1]-nose_height / 2)
        bottom_right = int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2)

        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)
        frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width] = final_nose


    cv2.imshow("Frame", frame)
    if nose_area is not None:
        cv2.imshow("nose", nose_area)

    key = cv2.waitKey(1)
    if key == 27:
        break