import cv2
import numpy

import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


video = cv2.VideoCapture('sample.mp4')
print(video.isOpened()) # ファイルを開けたか

numpy_frame_from_opencv = []
i = 0
while True:
    ret, frame = video.read()
    if ret == False:
        break
    i+=1
    if i%5 == 0:
        numpy_frame_from_opencv.append(frame)

video.release()


model_path = "./mediapipe_tasks/pose_landmarker_lite.task"



# Create a pose landmarker instance with the video mode:
options = mediapipe.tasks.vision.PoseLandmarkerOptions(
    base_options=mediapipe.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mediapipe.tasks.vision.RunningMode.VIDEO
)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = numpy.copy(rgb_image)

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

# Initialize landmarker
with mediapipe.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    
    annotated_images = []

    for i, img in enumerate(numpy_frame_from_opencv):
        mp_image = mediapipe.Image(
            image_format=mediapipe.ImageFormat.SRGB,
            data=img
        )
        pose_landmarker_result = landmarker.detect_for_video(mp_image, i)

        annotated_images.append(draw_landmarks_on_image(img.view(), pose_landmarker_result))
    
    for img in annotated_images:
        cv2.imshow("test", img)
        cv2.waitKey(100)