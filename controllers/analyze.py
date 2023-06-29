import cv2
import numpy

import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "./mediapipe_tasks/pose_landmarker_lite.task"


mp_image = mediapipe.Image(
    image_format=mediapipe.ImageFormat.SRGB,
    data=numpy_frame_from_opencv
)


# Create a pose landmarker instance with the video mode:
options = mediapipe.tasks.vision.PoseLandmarkerOptions(
    base_options=mediapipe.tasks.BaseOptions(model_asset_path=model_path),
    runnning_mode=VisionRunningMode.VIDEO
)


# Initialize landmarker
with mediapipe.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    print("hello")