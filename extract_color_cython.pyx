import cv2
import numpy as np
cimport numpy as cnp


def extract_main_colors_cython(str image_path, int n=6):
    cdef int panel_h = 64
    cdef int panel_w = 512

    cdef cnp.ndarray[cnp.uint8_t, ndim=3] image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    cdef cnp.ndarray[cnp.uint32_t, ndim=2] color_source = (image[image[:, :, -1] != 0, :3]).astype(np.uint32)

    cdef cnp.ndarray[cnp.uint32_t, ndim=1] packed_image = (color_source[:, 0] * 1e6 + color_source[:, 1] * 1e3 + color_source[:, 2]).astype(np.uint32)
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] unique_colors
    cdef cnp.ndarray[cnp.int64_t, ndim=1] pixel_counts
    unique_colors, pixel_counts = np.unique(packed_image, return_counts=True)
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] top_n_colors = unique_colors[np.argsort(pixel_counts)[::-1][:n]]

    cdef list color_panels = list()
    cdef unsigned int color
    cdef int b, g, r
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] color_panel
    for color in top_n_colors:
        b, g, r = int(color // 1e6), int(color // 1e3 % 1e3), int(color % 1e3)
        color_panel = np.ones((panel_h, panel_w, 3), np.uint8) * np.array((b, g, r), np.uint8)
        color_panels.append(color_panel)

    return color_panels
