import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

model_path = 'pose_landmarker_lite.task'

#numpy_image = np.array(cv2.imread('test.jpg'))

#mp_image = mp.Image.create_from_file('test.jpg')

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(cv2.imread('test.jpg')))

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

detector = vision.PoseLandmarker.create_from_options(options)

detection_result = detector.detect(mp_image)


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image



with PoseLandmarker.create_from_options(options) as landmarker:
    pose_landmarker_result = landmarker.detect(mp_image)
    print(pose_landmarker_result, end='\n')

    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    #cv2.imshow('test', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('test', cv2.resize(annotated_image, None, None, 0.9, 0.9))
    cv2.waitKey(60000)
