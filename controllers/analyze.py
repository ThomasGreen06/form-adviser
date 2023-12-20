import math
import cv2

# from rembg import remove
import mediapipe as mp
import numpy as np

# mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic(
#         static_image_mode=True,
#         min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils


class Color:
    RED = (0, 0, 255)
    GREEN = (0, 128, 0)
    BLUE = (255, 0, 0)


class Img:
    __mp_holistic = mp.solutions.holistic
    __mp_drawing = mp.solutions.drawing_utils

    def __init__(self, img_origin, mark_color):
        self.__results = self.__process_img(img_origin)
        self.__annotated_img = self.__draw_landmarks(img_origin, mark_color)
        self.__cropped_img = self.__crop_img()
        (self.__base_x, self.__base_y) = self.__get_basing_point()

        self.img = self.__cropped_img

    def width(self):
        return self.img.shape[1]

    def height(self):
        return self.img.shape[0]

    def __process_img(self, img):
        return self.__mp_holistic.Holistic(
            static_image_mode=True, min_detection_confidence=0.5
        ).process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __draw_landmarks(self, img, color):
        annotated_img = img
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_img,
            landmark_list=self.__results.pose_landmarks,
            connections=self.__mp_holistic.POSE_CONNECTIONS,
            connection_drawing_spec=self.__mp_drawing.DrawingSpec(
                color=color, thickness=5, circle_radius=5
            ),
        )
        return annotated_img

    def __crop_img(self):
        def format(ratio, index):
            return math.floor(ratio * self.__annotated_img.shape[index])

        landmark = self.__results.pose_landmarks.landmark
        margin = math.sqrt(
                math.pow(abs(landmark[13].x - landmark[15].x) *
                         self.__annotated_img.shape[1], 2) +
                math.pow(abs(landmark[13].y - landmark[15].y) *
                         self.__annotated_img.shape[0], 2))
        margin = math.floor(margin)

        landmark = self.__results.pose_landmarks.landmark
        max_j = format(max([elm.x for elm in landmark]), 1) + margin
        min_j = format(min([elm.x for elm in landmark]), 1) - margin
        max_i = format(max([elm.y for elm in landmark]), 0) + margin
        min_i = format(min([elm.y for elm in landmark]), 0) - margin

        return self.__annotated_img[min_i:max_i, min_j:max_j]

    def __get_basing_point(self):
        return (
            self.__results.pose_landmarks.landmark[11].x,
            self.__results.pose_landmarks.landmark[11].y,
        )


def analyze(origin_img1: np.ndarray, origin_img2: np.ndarray):
    img1 = Img(origin_img1, Color.RED)
    img2 = Img(origin_img2, Color.BLUE)

    if img1.width() < img2.width():
        ratio = img1.width() / img2.width()
        img2.img = cv2.resize(img2.img, dsize=None, fx=ratio, fy=ratio,
                              interpolation=cv2.INTER_CUBIC)
    else:
        ratio = img2.width() / img1.width()
        img1.img = cv2.resize(img1.img, dsize=None, fx=ratio, fy=ratio,
                              interpolation=cv2.INTER_CUBIC)

    print("img1: " + str(img1.img.shape))
    print("img2: " + str(img2.img.shape), end="\n\n")

    if img1.height() < img2.height():
        img2.img = img2.img[0:img1.height(), 0:img2.width()]
    else:
        img1.img = img1.img[0:img2.height(), 0:img1.width()]

    print("img1: " + str(img1.img.shape))
    print("img2: " + str(img2.img.shape))

    img1.img = cv2.addWeighted(img1.img, 0.5, img2.img, 0.5, 0)

    return img1.img

    #    image_origin = cv2_img

    #     p11 = Img(
    #         results.pose_landmarks.landmark[11].x * annotated_image.shape[1],
    #         results.pose_landmarks.landmark[11].y * annotated_image.shape[0],
    #     )
    #
    #     p12 = Img(
    #         results.pose_landmarks.landmark[12].x * annotated_image.shape[1],
    #         results.pose_landmarks.landmark[12].y * annotated_image.shape[0],
    #     )
    #
    #     results = holistic.process(
    #             cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB))
    #
    #     annotated_image = image_origin.copy()
    #
    #     mp_drawing.draw_landmarks(
    #             image=annotated_image,
    #             landmark_list=results.pose_landmarks,
    #             connections=mp_holistic.POSE_CONNECTIONS,
    #             connection_drawing_spec=mp_drawing.DrawingSpec(
    #                 color=BLUE,
    #                 thickness=5,
    #                 circle_radius=5))
    #
    # shoulder_width = math.sqrt(
    #     (abs(p11.x - p12.x)) ^ 2 + (abs(p11.y - p12.y)) ^ 2
    # )
    # print("shoulder_width: " + str(shoulder_width))
    #
    # model_annotated_image = np.zeros_like(model_img)
    # model_results = holistic.process(
    #     cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
    # )
    #
    # mp_drawing.draw_landmarks(
    #     image=model_annotated_image,
    #     landmark_list=model_results.pose_landmarks,
    #     connections=mp_holistic.POSE_CONNECTIONS,
    #     connection_drawing_spec=mp_drawing.DrawingSpec(
    #         color=RED, thickness=5, circle_radius=5
    #     ),
    # )
    #
    # p11_model = Img(
    #     results.pose_landmarks.landmark[11].x * \
    #        model_annotated_image.shape[1],
    #     results.pose_landmarks.landmark[11].y * \
    #        model_annotated_image.shape[0],
    # )
    #
    # p12_model = Img(
    #     results.pose_landmarks.landmark[12].x * \
    #        model_annotated_image.shape[1],
    #     results.pose_landmarks.landmark[12].y * \
    #        model_annotated_image.shape[0],
    # )
    #
    # shoulder_width_model = math.sqrt(
    #     (abs(p11_model.x - p12_model.x))
    #     ^ 2 + (abs(p11_model.y - p12_model.y))
    #     ^ 2
    # )
    # print(shoulder_width_model)
    #
    # if shoulder_width < shoulder_width_model:
    #     ratio = shoulder_width / shoulder_width_model
    #     cv2.resize(
    #         model_annotated_image,
    #         dsize=None,
    #         fx=ratio,
    #         fy=ratio,
    #         interpolation=cv2.INTER_LANCZOS4,
    #     )
    # else:
    #     ratio = shoulder_width_model / shoulder_width
    #     cv2.resize(
    #         annotated_image,
    #         dsize=None,
    #         fx=ratio,
    #         fy=ratio,
    #         interpolation=cv2.INTER_LANCZOS4,
    #     )
    #
    # # annotated_image = remove(annotated_image)
    # annotated_image = annotated_image.copy()
    # # annotated_image[0:800][0:100] = np.array([66, 245, 75, 1])
    # print(annotated_image.shape)
    # print(model_annotated_image.shape)
    #
    # col = (
    #     annotated_image.shape[0]
    #     if annotated_image.shape[0] < model_annotated_image.shape[0]
    #     else model_annotated_image.shape[0]
    # )
    #
    # low = (
    #     annotated_image.shape[1]
    #     if annotated_image.shape[1] < model_annotated_image.shape[1]
    #     else model_annotated_image.shape[1]
    # )
    #
    # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGBA2RGB)
    #
    # annotated_image[0:col][0:low] = (
    #     annotated_image[0:col][0:low] * 0.5
    #     + model_annotated_image[0:col][0:low] * 0.5
    # )
    #
    # return annotated_image
