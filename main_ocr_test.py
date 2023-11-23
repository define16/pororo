import os.path
import os
import time

import cv2
from matplotlib import pyplot as plt


import cv2
from pororo import Pororo
from pororo.pororo import SUPPORTED_TASKS
from utils.image_util import plt_imshow, put_text
import warnings

warnings.filterwarnings('ignore')

def clustering(image):
    count = 0
    free_space = []
    before_min_dot = 0
    min_dots, max_dots = [], []
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i in range(1, len(contours)):
        r = cv2.boundingRect(contours[i])
        min_dots.append(r[1])
        max_dots.append(r[1] + r[3])
    min_dots = sorted(list(set(min_dots)))
    max_dots = sorted(list(set(max_dots)))

    for max_dot in max_dots:
        if count == 0:
            count += 1
            continue
        for min_dot in min_dots:
            if max_dot < min_dot:
                if before_min_dot < min_dot:
                    if free_space and free_space[-1][1] == before_min_dot:
                        free_space[-1] = (max_dot, free_space[-1][1])
                    else:
                        free_space.append((max_dot, before_min_dot))
                break
            if min_dot < before_min_dot:
                continue
            before_min_dot = min_dot

    new_free_space = []
    before = (-1, -1)
    for fs in free_space:
        if fs[0] - before[1] < 200 and before[1] >= 0:
            new_free_space[-1] = (fs[0], new_free_space[-1][1])
        else:
            new_free_space.append(fs)
        before = fs
    return new_free_space


def divide_image(image):
    _, thresh = cv2.threshold(image, 227, 255, cv2.THRESH_TOZERO)
    thresh_image = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
    chunk_coordination = []
    if thresh_image.shape[1] + 200 < thresh_image.shape[0]:
        free_space = clustering(thresh)
        before_y_axis = 0
        for nfs in free_space:
            chunk_coordination.append(((0, before_y_axis), (thresh_image.shape[1], nfs[0] + 10)))
            before_y_axis = nfs[0]
        if before_y_axis != thresh_image.shape[0]:
            chunk_coordination.append(((0, before_y_axis), (thresh_image.shape[1], thresh_image.shape[0])))
    else:
        chunk_coordination.append(((0, 0), (thresh_image.shape[1], thresh_image.shape[0])))
    return [image[cc[0][1]:cc[1][1], cc[0][0]:cc[1][0]] for cc in chunk_coordination]


class PororoOcr:
    def __init__(self, model: str = "brainocr", lang: str = "ko", **kwargs):
        self.model = model
        self.lang = lang
        self._ocr = Pororo(task="ocr", lang=lang, model=model, **kwargs)
        self.img_path = None
        self.ocr_result = {}

    def run_ocr(self, img_path: str, debug: bool = False):
        self.img_path = img_path
        self.ocr_result = self._ocr(img_path, detail=True)

        if self.ocr_result['description']:
            ocr_text = self.ocr_result["description"]
        else:
            ocr_text = "No text detected."

        if debug:
            self.show_img_with_ocr()

        return ocr_text

    @staticmethod
    def get_available_langs():
        return SUPPORTED_TASKS["ocr"].get_available_langs()

    @staticmethod
    def get_available_models():
        return SUPPORTED_TASKS["ocr"].get_available_models()

    def get_ocr_result(self):
        return self.ocr_result

    def get_img_path(self):
        return self.img_path

    def show_img(self):
        plt_imshow(img=self.img_path)

    def show_img_with_ocr(self):
        img = cv2.imread(self.img_path)
        roi_img = img.copy()

        for text_result in self.ocr_result['bounding_poly']:
            text = text_result['description']
            tlX = text_result['vertices'][0]['x']
            tlY = text_result['vertices'][0]['y']
            trX = text_result['vertices'][1]['x']
            trY = text_result['vertices'][1]['y']
            brX = text_result['vertices'][2]['x']
            brY = text_result['vertices'][2]['y']
            blX = text_result['vertices'][3]['x']
            blY = text_result['vertices'][3]['y']

            pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))

            topLeft = pts[0]
            topRight = pts[1]
            bottomRight = pts[2]
            bottomLeft = pts[3]

            cv2.line(roi_img, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(roi_img, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(roi_img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(roi_img, bottomLeft, topLeft, (0, 255, 0), 2)
            roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 20, font_size=15)

            # print(text)

        plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(50, 50))


if __name__ == "__main__":
    ocr = PororoOcr()
    image_path = os.path.join("dataset", "aaa.png")
    text = ocr.run_ocr(image_path, debug=True)
    print('Result :', text)

    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(image_gray.shape)

    # chunk_images = divide_image(image_gray)
    # for chunk_image in chunk_images:
    #     text = ocr.run_ocr(image_path, debug=True)

