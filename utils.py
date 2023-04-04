# Create OCR dummy datasets: image and bounding boxes
import numpy as np


def create_image(
    height: int,
    width: int,
    channel: int = 3,
    seed: int = 42,
) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(0, 256, (height, width, channel), np.uint8)


def create_bboxes(
    num_bboxes: int,
    max_height: int,
    max_width: int,
    safety_zone_ratio: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    np.random.seed(seed)

    # 일반 문서처럼 페이지 여백을 주기 위한 설정
    boundary = {
        "left": int(max_width * safety_zone_ratio),
        "right": int(max_width - max_width * safety_zone_ratio),
        "top": int(max_height * safety_zone_ratio),
        "bottom": int(max_height - max_height * safety_zone_ratio),
    }

    _h = 20  # bbox 높이
    _w = 80  # bbox 너비
    _h_gap = 10  # bbox 사이 세로 간격
    _w_gap = 15  # bbox 사이 가로 간격

    cum_x = boundary["left"]
    cum_y = boundary["top"]

    bboxes = [list() for _ in range(num_bboxes)]

    for idx in range(num_bboxes):
        lt = [cum_x, cum_y]
        rt = [cum_x + _w, cum_y]
        rb = [cum_x + _w, cum_y + _h]
        lb = [cum_x, cum_y + _h]

        angle = np.random.uniform(-3, 3)  # 랜덤한 bbox 회전 각도
        angle_rad = np.deg2rad(angle)

        # rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        center = np.array([cum_x + _w / 2, cum_y + _h / 2])
        points = np.array([lt, rt, rb, lb]) - center

        # bbox 회전
        rotated_points = np.dot(points, rotation_matrix.T) + center
        bboxes[idx] = rotated_points

        cum_x = rt[0] + _w_gap

        if cum_x + _w > boundary["right"]:
            cum_x = boundary["left"]
            cum_y += _h + _h_gap

    return np.array(bboxes, dtype=np.float32)
