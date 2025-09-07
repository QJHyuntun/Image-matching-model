import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# === 导入GlueStick相关模块 ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.models.two_view_pipeline import TwoViewPipeline

# === 导入XFeat模块 ===
from Xfeat_modules.xfeat import XFeat


def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=3):
    """在图像上绘制关键点"""
    if keypoints is None:
        return image
    for pt in keypoints:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(image, (x, y), radius, color, -1)
    return image


def draw_lines(image, lines, color=(255, 0, 0), thickness=2):
    """在图像上绘制线段"""
    if lines is None:
        return image
    for line in lines:
        pt1 = tuple(map(int, line[0]))
        pt2 = tuple(map(int, line[1]))
        cv2.line(image, pt1, pt2, color, thickness)
    return image


def main(image1_path, image2_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # === 读取图像 ===
    img1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"无法读取图像，请检查路径:\n{image1_path}\n{image2_path}")

    # 灰度图并确保是标准np.ndarray
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = np.asarray(gray1, dtype=np.uint8)
    gray2 = np.asarray(gray2, dtype=np.uint8)

    # === 初始化GlueStick ===
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {'force_num_keypoints': False, 'max_num_keypoints': 1000},
            'wireframe_params': {'merge_points': True, 'merge_line_endpoints': True},
            'max_n_lines': 400,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': str(GLUESTICK_ROOT / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {'from_pose_depth': False}
    }
    pipeline_model = TwoViewPipeline(conf).to(device).eval()

    # === 转换为Torch张量 ===
    # torch_gray1 = numpy_image_to_torch(np.array(gray1, copy=False)).to(device)[None]
    torch_gray1 = numpy_image_to_torch(np.array(gray1, dtype=np.float32)).to(device)[None]

    # torch_gray2 = numpy_image_to_torch(np.array(gray2, copy=False)).to(device)[None]
    torch_gray2 = numpy_image_to_torch(np.array(gray1, dtype=np.float32)).to(device)[None]

    # === GlueStick特征提取 ===
    pred = pipeline_model({'image0': torch_gray1, 'image1': torch_gray2})
    pred = batch_to_np(pred)

    kp1, kp2 = pred.get('keypoints0', []), pred.get('keypoints1', [])
    lines1, lines2 = pred.get('lines0', []), pred.get('lines1', [])

    # === 绘制GlueStick特征 ===
    img1_glue = draw_keypoints(img1.copy(), kp1, color=(0, 255, 0))
    img1_glue = draw_lines(img1_glue, lines1, color=(255, 0, 0))

    img2_glue = draw_keypoints(img2.copy(), kp2, color=(0, 255, 0))
    img2_glue = draw_lines(img2_glue, lines2, color=(255, 0, 0))

    # === XFeat特征提取 ===
    xfeat = XFeat()
    mkpts0, mkpts1 = xfeat.match_xfeat(img1, img2, top_k=500)

    img1_xfeat = draw_keypoints(img1.copy(), mkpts0, color=(0, 0, 255))
    img2_xfeat = draw_keypoints(img2.copy(), mkpts1, color=(0, 0, 255))

    # === 拼接显示结果 ===
    top_glue = np.hstack((img1_glue, img2_glue))
    top_xfeat = np.hstack((img1_xfeat, img2_xfeat))
    final = np.vstack((top_glue, top_xfeat))

    cv2.imshow("GlueStick (top) and XFeat (bottom) features", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img1_path = r"E:\Image-matching-model\GS_Xfeat-main\resources\img1.jpg"
    img2_path = r"E:\Image-matching-model\GS_Xfeat-main\resources\img2.jpg"
    main(img1_path, img2_path)
