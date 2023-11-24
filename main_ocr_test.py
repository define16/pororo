import os
import time

import cv2
from pororo.models.brainOCR import brainocr
from pororo.tasks import download_or_load
from pororo.tasks.utils.base import TaskConfig, PororoSimpleBase
from utils.image_util import plt_imshow, put_text
import warnings

warnings.filterwarnings('ignore')


def show_img_with_ocr(image, ocr_result):
    roi_img = image.copy()

    for text_result in ocr_result['bounding_poly']:
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

    plt_imshow(["Original", "ROI"], [image, roi_img], figsize=(50, 50))


class PororoOCR(PororoSimpleBase):
    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def _postprocess(self, ocr_results, detail: bool = False):
        """
        Post-process for OCR result

        Args:
            ocr_results (list): list contains result of OCR
            detail (bool): if True, returned to include details. (bounding poly, vertices, etc)

        """
        sorted_ocr_results = sorted(
            ocr_results,
            key=lambda x: (
                x[0][0][1],
                x[0][0][0],
            ),
        )

        if not detail:
            return [
                sorted_ocr_results[i][-1]
                for i in range(len(sorted_ocr_results))
            ]

        result_dict = {
            "description": list(),
            "bounding_poly": list(),
        }

        for ocr_result in sorted_ocr_results:
            vertices = list()

            for vertice in ocr_result[0]:
                vertices.append({
                    "x": vertice[0],
                    "y": vertice[1],
                })

            result_dict["description"].append(ocr_result[1])
            result_dict["bounding_poly"].append({
                "description": ocr_result[1],
                "vertices": vertices
            })

        return result_dict

    def predict(self, image, **kwargs):
        """
        Conduct Optical Character Recognition (OCR)

        Args:
            image_path (str): the image file path
            detail (bool): if True, returned to include details. (bounding poly, vertices, etc)

        """
        detail = kwargs.get("detail", False)

        return self._postprocess(
            self._model(
                image,
                skip_details=False,
                batch_size=1,
                paragraph=True,
            ),
            detail,
        )


class Ocr:
    def __init__(self, n_model: str = "brainocr", lang: str = "ko"):
        self._ocr_result = {}
        available_models = {"ko": ["brainbert.base.ko.korquad"]}
        if lang not in {"ko"}:
            raise ValueError(
                f"Unsupported Language : {lang}",
                'Support Languages : ["en", "ko"]',
            )
        self._n_model = n_model
        self._lang = lang
        self._detect_model = "craft"
        self._ocr_opt = "ocr-opt"

        device = "cpu"  # "cpu or "cuda"
        config = TaskConfig(
            task="ocr",
            lang="ko",
            n_model=n_model
        )
        det_model_path = download_or_load(
            f"misc/{self._detect_model}.pt",
            config.lang,
        )
        rec_model_path = download_or_load(
            f"misc/{config.n_model}.pt",
            config.lang,
        )
        opt_fp = download_or_load(
            f"misc/{self._ocr_opt}.txt",
            config.lang,
        )
        model = brainocr.Reader(
            config.lang,
            det_model_ckpt_fp=det_model_path,
            rec_model_ckpt_fp=rec_model_path,
            opt_fp=opt_fp,
            device=device,
        )
        model.detector.to(device)
        model.recognizer.to(device)
        self._model = model
        self._ocr = PororoOCR(model=model, config=config)

    def run_ocr(self, image_path: str, debug: bool = False):
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(image_gray.shape)
        chunk_images = self.divide_image(image_gray)
        print(f"{len(chunk_images)}로 이미지 분리")
        ocr_text = []
        if len(chunk_images) > 1:
            for c_img in chunk_images:
                _ocr_result = self._ocr.predict(c_img, detail=True)
                print("aaaaaaaaaaa")
                print(_ocr_result)
                if _ocr_result['description']:
                    ocr_text.extend(_ocr_result["description"])
                else:
                    ocr_text.append("No text detected.")
        else:
            _ocr_result = self._ocr.predict(image_gray, detail=True)
            if _ocr_result['description']:
                ocr_text.extend(_ocr_result["description"])
            else:
                ocr_text.append("No text detected.")

            if debug:
                show_img_with_ocr(image_gray, _ocr_result)

        return ocr_text

    def clustering(self, image):
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

    def divide_image(self, image):
        _, thresh = cv2.threshold(image, 227, 255, cv2.THRESH_TOZERO)
        thresh_image = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        chunk_coordination = []
        if thresh_image.shape[1] + 200 < thresh_image.shape[0]:
            free_space = self.clustering(thresh)
            before_y_axis = 0
            for nfs in free_space:
                chunk_coordination.append(((0, before_y_axis), (thresh_image.shape[1], nfs[0] + 10)))
                before_y_axis = nfs[0]
            if before_y_axis != thresh_image.shape[0]:
                chunk_coordination.append(((0, before_y_axis), (thresh_image.shape[1], thresh_image.shape[0])))
        else:
            chunk_coordination.append(((0, 0), (thresh_image.shape[1], thresh_image.shape[0])))
        return [image[cc[0][1]:cc[1][1], cc[0][0]:cc[1][0]] for cc in chunk_coordination]


if __name__ == "__main__":
    start = time.time()
    ocr = Ocr()
    print(f"P1 : {time.time() - start}")
    image_path = os.path.join("dataset", "img_1.png")
    text = ocr.run_ocr(image_path, debug=True)
    print(time.time()-start)
    print('Result :', text)


