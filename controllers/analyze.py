import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2

MODEL_PATH = 'controllers/mediapipe_tasks/pose_landmarker_lite.task'
VisionRunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE)

def draw_landmarks_on_image(rgb_image, mp_img):
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    detector = vision.PoseLandmarker.create_from_options(options)
    detection_result = detector.detect(mp_img)

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


def analyze(cv2_img: np.ndarray):  # cv2_img = cv2.imread('File_Name')

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(cv2_img))

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(mp_img)
        #print(pose_landmarker_result, end='\n')
        print("###### FUNC 'analyze' SUCCESSED !! #######", end='\n\n')
        annotated_image = draw_landmarks_on_image(mp_img.numpy_view(), mp_img)
        return annotated_image



#analyze(cv2.imread('test.jpg'))