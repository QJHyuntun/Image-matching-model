import argparse
import os
from datetime import datetime

import onnx
import onnxsim
import torch
import numpy as np
from pytlsd import lsd
import cv2

from gluestick.models.two_view_pipeline_tp import TwoViewPipeline


def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float().to('cuda')[None]


def detect_lsd_lines(img, min_length=5, max_n_lines=400, device=None):
    lines, scores = [], []
    b_segs = lsd(img)
    segs_length = np.linalg.norm(b_segs[:, 2:4] - b_segs[:, 0:2], axis=1)
    # Remove short lines
    b_segs = b_segs[segs_length >= min_length]
    segs_length = segs_length[segs_length >= min_length]
    b_scores = b_segs[:, -1] * np.sqrt(segs_length)
    # Take the most relevant segments with
    indices = np.argsort(-b_scores)
    indices = indices[:max_n_lines]
    lines.append(torch.from_numpy(b_segs[indices, :4].reshape(-1, 2, 2)))
    scores.append(torch.from_numpy(b_scores[indices]))

    lines = torch.stack(lines).to(device)
    scores = torch.stack(scores).to(device)*100

    # 扩展
    if lines.size(1) < max_n_lines:
        lpadding = (0, 0, 0, 0, 0, max_n_lines - lines.size(1))  # (left, right, top, bottom)
        spadding = (0, max_n_lines - scores.size(1))
        lines = torch.nn.functional.pad(lines, lpadding, value=0)
        scores = torch.nn.functional.pad(scores, spadding, value=0)

    return lines, scores


def AddAuxiliaryLine(img):
    # 检测图像维度
    if img.ndim == 3:  # 彩色图像
        # 假设彩色图像的最后一个维度是颜色通道
        img[12, 101:182, :] = np.array([255, 255, 255], dtype=img.dtype)
        img[13, 101:182, :] = np.array([255, 255, 255], dtype=img.dtype)
    elif img.ndim == 2:  # 灰度图像
        img[12, 101:182] = 255  # 为了与彩色图像的亮度保持一致，也使用相对于255的值
        img[13, 101:182] = 255
    else:
        raise ValueError("Unsupported image format")


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TwoViewPipeline().to(device).eval()

    # img1 = np.random.randint(0, 256, size=(1024, 1024), dtype=np.uint8)
    # img2 = np.random.randint(0, 256, size=(1920, 1080), dtype=np.uint8)

    # img1Path = "/home/afsy/0_Datasets/ImageMatch/20240312/uav-visible/dji_zt02_1.jpg"
    # img2Path = "/home/afsy/0_Datasets/ImageMatch/20240312/targets/ir/zt01-45.bmp"

    img1Path = "/home/sy/0_Datasets/Images/ImageMatch/20240312/sat-visible/sat-zt03_1.jpg"
    img2Path = "/home/sy/0_Datasets/Images/ImageMatch/20240312/targets/vis/zt02-45-hk.bmp"

    img1 = cv2.imread(img1Path, 0)
    img2 = cv2.imread(img2Path, 0)

    AddAuxiliaryLine(img1)
    l1, s1 = detect_lsd_lines(img1, device=device)  # lines detection
    AddAuxiliaryLine(img2)
    l2, s2 = detect_lsd_lines(img2, device=device)  # lines detection

    # save lines and scores to npy files
    np_lines = l1.cpu().numpy()
    np_scores = s1.cpu().numpy()
    np.save('onnx/lines.npy', np_lines)
    np.save('onnx/scores.npy', np_scores)

    torchImg1, torchImg2 = numpy_image_to_torch(img1), numpy_image_to_torch(img2)

    out_path_ori = "onnx/SN_Matcher_ori.onnx"
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": 1. start export onnx... ")
    torch.onnx.export(
        model,
        (torchImg1, l1, s1, torchImg2, l2, s2),
        out_path_ori,
        export_params=True,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=13,
        input_names=['img1', 'inLines1', 'inScores1', 'img2', 'inLines2', 'inScores2'],
        output_names=['kpts1', 'kpts2', 'mkpts', 'mkptsScores', 'lines1', 'lines2', 'mlines', 'mlinesScores'],
        dynamic_axes=
        {
            # input
            # 'img1': {2: 'height', 3: 'width'},
            # 'inLines1': {1: 'count'},
            # 'inScores1': {1: 'scores'},
            # 'img2': {2: 'height', 3: 'width'},
            # 'inLines2': {1: 'count'},
            # 'inScores2': {1: 'scores'},
            # output
            'kpts1': {1: 'count'},
            'kpts2': {1: 'count'},
            'mkpts': {1: 'count'},
            # 'mkptsScores': {1: 'scores'},
            'lines1': {1: 'count'},
            'lines2': {1: 'count'},
            'mlines': {1: 'count'},
            'mlinesScores': {1: 'scores'}
        }
    )
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": onnx file exported.")

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": check onnx file...")
    # Checks
    model_onnx = onnx.load(out_path_ori)  # load onnx model
    try:
        onnx.checker.check_model(model_onnx)  # check onnx model
    except Exception:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": Model incorrect.")
    else:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": Model correct.")

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": onnx file is done. ")

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": 2. start simplifying onnx... ")
    out_path_sim = "onnx/SN_Matcher_sim.onnx"
    model_sim, flag = onnxsim.simplify(out_path_ori)
    if flag:
        onnx.save(model_sim, out_path_sim)
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": simplify onnx successfully.")
    else:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": simplify onnx failed.")

    # Modify onnx, rename duplicated node names
    # ==================================遍历节点，命名没有名称的节点，重新保存模型======================================
    # 遍历所有节点
    modify_flag = False
    node_names = []
    prefix = ''
    count = 0
    for idx, node in enumerate(model_sim.graph.node):
        # 命名没名称的节点
        if not node.name:
            node.name = prefix+f'/unnamed_node_{idx}'
            modify_flag = True
            count += 1
        node_names.append(node.name)
        slash_pos = node.name.rfind('/')
        prefix = node.name[:slash_pos] if slash_pos != -1 else ''
        print(f'{idx}:{node.name}')

    # save
    with open('onnx/onnx_node_name.txt', 'w') as file:
        file.write('\n'.join(node_names))

    if modify_flag:
        print(f'\nThere is(are) {count} unnamed node(s).')
        onnx.save(model_sim, 'onnx/SN_Matcher_sim.onnx')

    print("Model saved to:", os.getcwd()+'/'+out_path_ori +' and ' + out_path_sim)
