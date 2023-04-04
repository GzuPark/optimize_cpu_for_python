# line profiler를 실행하고 싶다면, 적절하게 주석을 지우고 실행해주세요.
import os
from time import perf_counter

import cv2
import numpy as np
# from line_profiler_pycharm import profile

from extract_color_cython import extract_main_colors_cython


# @profile
def extract_main_colors_naive(image_path: str, n: int = 6) -> np.ndarray:
    panel_h = 64
    panel_w = 512

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    color_source = (image[image[:, :, -1] != 0, :3]).astype(np.uint32)

    packed_image = color_source[:, 0] * 1e6 + color_source[:, 1] * 1e3 + color_source[:, 2]
    unique_colors, pixel_counts = np.unique(packed_image, return_counts=True)
    top_n_colors = unique_colors[np.argsort(pixel_counts)[::-1][:n]]

    color_panels = list()
    for color in top_n_colors:
        b, g, r = color//1e6, color//1e3 % 1e3, color % 1e3
        color_panel = np.ones((panel_h, panel_w, 3)) * (b, g, r)
        color_panels.append(color_panel)

    color_panels = np.concatenate(color_panels, axis=0)

    return color_panels.astype(np.uint8)


# @profile
def extract_main_colors_fast(image_path: str, n: int = 6) -> np.ndarray:
    panel_h = 64
    panel_w = 512

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    color_source = (image[image[:, :, -1] != 0, :3]).astype(np.uint32)

    packed_image = np.left_shift(color_source[:, 0], 16) + np.left_shift(color_source[:, 1], 8) + color_source[:, 2]
    unique_colors, pixel_counts = np.unique(packed_image, return_counts=True)
    top_n_colors = unique_colors[np.argsort(pixel_counts)[::-1][:n]]

    color_panels = np.empty((panel_h * n, panel_w, 3), np.uint8)
    R = np.bitwise_and(top_n_colors, 0xff)
    G = np.right_shift(np.bitwise_and(top_n_colors, 0xff00), 8)
    B = np.right_shift(np.bitwise_and(top_n_colors, 0xff0000), 16)
    for idx, (b, g, r) in enumerate(zip(B, G, R)):
        color_panels[idx * panel_h: (idx+1) * panel_h, :, 0].fill(b)
        color_panels[idx * panel_h: (idx+1) * panel_h, :, 1].fill(g)
        color_panels[idx * panel_h: (idx+1) * panel_h, :, 2].fill(r)

    return color_panels


# @profile
def extract_main_colors_faster(image_path: str, n: int = 6) -> np.ndarray:
    panel_h = 64
    panel_w = 512

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    roi = image[:, :, -1].ravel() != 0

    packed_image = np.left_shift(image[:, :, 0].ravel().astype(np.uint32)[roi], 16) + np.left_shift(image[:, :, 1].ravel().astype(np.uint32)[roi], 8) + image[:, :, 2].ravel().astype(np.uint32)[roi]
    unique_colors, pixel_counts = np.unique(packed_image, return_counts=True)
    top_n_colors = unique_colors[np.argsort(pixel_counts)[::-1][:n]]

    color_panels = np.empty((panel_h * n, panel_w, 3), np.uint8)
    R = np.bitwise_and(top_n_colors, 0xff)
    G = np.right_shift(np.bitwise_and(top_n_colors, 0xff00), 8)
    B = np.right_shift(np.bitwise_and(top_n_colors, 0xff0000), 16)
    for idx, (b, g, r) in enumerate(zip(B, G, R)):
        color_panels[idx * panel_h: (idx+1) * panel_h, :, 0].fill(b)
        color_panels[idx * panel_h: (idx+1) * panel_h, :, 1].fill(g)
        color_panels[idx * panel_h: (idx+1) * panel_h, :, 2].fill(r)

    return color_panels


def main():
    image_path = os.path.join("assets", "image.png")

    st = perf_counter()
    for _ in range(100):
        _ = extract_main_colors_naive(image_path)
    print(f"Naive runs 100 times: {perf_counter() - st:.3f} second")

    st = perf_counter()
    for _ in range(100):
        _ = extract_main_colors_fast(image_path)
    print(f"Fast runs 100 times: {perf_counter() - st:.3f} second")

    st = perf_counter()
    for _ in range(100):
        _ = extract_main_colors_faster(image_path)
    print(f"Faster runs 100 times: {perf_counter() - st:.3f} second")

    st = perf_counter()
    for _ in range(100):
        _ = extract_main_colors_cython(image_path)  # no line profiler
    print(f"Cython runs 100 times: {perf_counter() - st:.3f} second")


if __name__ == "__main__":
    main()
