import math
import cv2
from rembg import remove
import mediapipe as mp
import numpy as np

RED = (0, 0, 255)
GREEN = (0, 128, 0)
BLUE = (255, 0, 0)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def analyze(cv2_img: np.ndarray, model_img: np.ndarray):
    image_origin = cv2_img
    results = holistic.process(
            cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB))

    annotated_image = image_origin.copy()

    mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=BLUE,
                thickness=5,
                circle_radius=5))

    class coord:
        def __init__(self, x, y):
            self.x = math.floor(x)
            self.y = math.floor(y)

    p11 = coord(
            results.pose_landmarks.landmark[11].x * annotated_image.shape[1],
            results.pose_landmarks.landmark[11].y * annotated_image.shape[0]
            )

    p12 = coord(
            results.pose_landmarks.landmark[12].x * annotated_image.shape[1],
            results.pose_landmarks.landmark[12].y * annotated_image.shape[0]
            )

    shoulder_width = math.sqrt((abs(p11.x - p12.x)) ^ 2
                               + (abs(p11.y - p12.y)) ^ 2)
    print("shoulder_width: " + str(shoulder_width))

    model_annotated_image = np.zeros_like(model_img)
    model_results = holistic.process(
            cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB))

    mp_drawing.draw_landmarks(
            image=model_annotated_image,
            landmark_list=model_results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=RED,
                thickness=5,
                circle_radius=5))

    p11_model = coord(
            results.pose_landmarks.landmark[11].x *
            model_annotated_image.shape[1],
            results.pose_landmarks.landmark[11].y *
            model_annotated_image.shape[0]
            )

    p12_model = coord(
            results.pose_landmarks.landmark[12].x *
            model_annotated_image.shape[1],
            results.pose_landmarks.landmark[12].y *
            model_annotated_image.shape[0]
            )

    shoulder_width_model = math.sqrt((abs(p11_model.x - p12_model.x)) ^ 2
                                     + (abs(p11_model.y - p12_model.y)) ^ 2)
    print(shoulder_width_model)

    return remove(annotated_image)
