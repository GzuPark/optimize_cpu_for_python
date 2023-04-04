from time import perf_counter
from typing import List

import cv2
import numpy as np

from utils import create_bboxes, create_image

import image_crop_module
import image_crop_module_omp


def crop(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    lt, rt, rb, lb = bbox
    width_a = np.sqrt(((rb[0] - lb[0]) ** 2) + ((rb[1] - lb[1]) ** 2))
    width_b = np.sqrt(((rt[0] - lt[0]) ** 2) + ((rt[1] - lt[1]) ** 2))
    height_a = np.sqrt(((rt[0] - rb[0]) ** 2) + ((rt[1] - rb[1]) ** 2))
    height_b = np.sqrt(((lt[0] - lb[0]) ** 2) + ((lt[1] - lb[1]) ** 2))
    max_width, max_height = max(int(width_a), int(width_b)), max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    m = cv2.getPerspectiveTransform(bbox.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped


def image_crop_python(image: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
    cropped_images = list()

    for bbox in bboxes:
        cropped_images.append(crop(image, bbox))

    return cropped_images


def image_crop_cpp(image: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
    return image_crop_module.image_crop(image, bboxes)


def image_crop_cpp_omp(image: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
    return image_crop_module_omp.image_crop(image, bboxes)


def main():
    random_seed = 42
    h = 3508
    w = 2480
    num_bboxes = 1528

    image = create_image(height=h, width=w, seed=random_seed)
    bboxes = create_bboxes(num_bboxes=num_bboxes, max_height=h, max_width=w, seed=random_seed)

    st = perf_counter()
    for _ in range(100):
        _ = image_crop_python(image, bboxes)
    print(f"Python runs 100 times: {perf_counter() - st:.3f} second")

    st = perf_counter()
    for _ in range(100):
        _ = image_crop_cpp(image, bboxes)
    print(f"C++ wrapper runs 100 times: {perf_counter() - st:.3f} second")

    st = perf_counter()
    for _ in range(100):
        _ = image_crop_cpp_omp(image, bboxes)
    print(f"OpenMP runs 100 times: {perf_counter() - st:.3f} second")


if __name__ == "__main__":
    main()
