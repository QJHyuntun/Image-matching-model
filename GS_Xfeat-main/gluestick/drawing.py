import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
from shapely.geometry import Point, Polygon


def DisplayMatchResults(image0, image1, objRect, refPt, estPt_sp, estPt_gs,
                        mPtsIdxInRect, mkpts0, mkpts1,
                        mLinesIdxInRect, mlines0, mlines1,
                        ptscolor, linecolor, text, path=None,
                        margin=10,
                        opencv_display=False, opencv_title='',
                        small_text=[]):
    cv2.rectangle(image0, objRect[0], objRect[2], (0, 255, 0), 3)
    # img = cv2.imread('E:\\Image-matching-model\\600.jpg')
    # ptos = [263, 361]
    # cv2.line(img, (ptos[0] - 25, ptos[1]), (ptos[0] + 25, ptos[1]), (0, 255, 0), 2)
    # cv2.line(img, (ptos[0], ptos[1] - 25), (ptos[0], ptos[1] + 25), (0, 255, 0), 2)
    cv2.line(image0, (refPt[0] - 25, refPt[1]), (refPt[0] + 25, refPt[1]), (0, 255, 0), 2)
    cv2.line(image0, (refPt[0], refPt[1] - 25), (refPt[0], refPt[1] + 25), (0, 255, 0), 2)

    # global m1tl, m1br
    global m1tl_pt, m1br_pt
    H0, W0, _ = image0.shape
    H1, W1, _ = image1.shape
    H, W = max(H0, H1), W0 + W1

    # 提取匹配点
    # if len(mPtsIdxInRect) >= 5:
    #     mk0 = [mkpts0[x] for x in mPtsIdxInRect]
    #     mk1 = [mkpts1[x] for x in mPtsIdxInRect]
    #     x0 = [int(point[0]) for point in mk0]
    #     y0 = [int(point[1]) for point in mk0]
    #     m0tl = min(x0), min(y0)
    #     m0br = max(x0), max(y0)
    #     x1 = [int(point[0]) for point in mk1]
    #     y1 = [int(point[1]) for point in mk1]
    #     m1tl_pt = min(x1), min(y1)
    #     m1br_pt = max(x1), max(y1)
    #     cv2.rectangle(image0, m0tl, m0br, (255, 0, 0), 3)
    #     cv2.rectangle(image1, m1tl_pt, m1br_pt, (255, 0, 0), 3)
    #
    # # 提取匹配线
    # if len(mLinesIdxInRect) >= 3:
    #     lines0 = [mlines0[x] for x in mLinesIdxInRect]
    #     pts0 = [point[0] for point in lines0]
    #     pts1 = [point[1] for point in lines0]
    #     pts = pts0 + pts1
    #     min_x, max_x = min([point[0] for point in pts]), max([point[0] for point in pts])
    #     min_y, max_y = min([point[1] for point in pts]), max([point[1] for point in pts])
    #     m0tl = int(min_x), int(min_y)
    #     m0br = int(max_x), int(max_y)
    #     lines1 = [mlines1[x] for x in mLinesIdxInRect]
    #     pts0 = [point[0] for point in lines1]
    #     pts1 = [point[1] for point in lines1]
    #     pts = pts0 + pts1
    #     min_x, max_x = min([point[0] for point in pts]), max([point[0] for point in pts])
    #     min_y, max_y = min([point[1] for point in pts]), max([point[1] for point in pts])
    #     m1tl = int(min_x), int(min_y)
    #     m1br = int(max_x), int(max_y)
    #     cv2.rectangle(image0, m0tl, m0br, (0, 0, 255), 3)
    #     cv2.rectangle(image1, m1tl, m1br, (0, 0, 255), 3)  # 绘制目标图像的红色矩形框

    draw = 0
    if draw:
        # draw matched lines
        mlines0, mlines1 = np.round(mlines0).astype(int), np.round(mlines1).astype(int)
        # mlines0 = [mlines0[x] for x in mLinesIdxInRect]
        # mlines1 = [mlines1[x] for x in mLinesIdxInRect]
        color = (np.array(linecolor[:, :3]) * 255).astype(int)[:, ::-1]
        for l0, l1, c in zip(mlines0, mlines1,color ):
            c = c.tolist()
            cv2.line(image0, (l0[0][0], l0[0][1]), (l0[1][0], l0[1][1]),
                     color=c, thickness=3)
            cv2.line(image1, (l1[0][0], l1[0][1]), (l1[1][0], l1[1][1]),
                     color=c, thickness=3)
        # cv2.imwrite('H:\\list\\img.jpg', image0)
        # cv2.imwrite('H:\\list\\img-1.jpg', image1)
    # draw reference point
    # cv2.circle(image0, refPt, 5, (0, 255, 0), -1)  # -1 表示填充圆
    if len(estPt_sp) > 0:
        cv2.circle(image1, estPt_sp, 5, (255, 0, 0), -1)
        cv2.line(image1, (estPt_sp[0] - 50, estPt_sp[1]), (estPt_sp[0] + 50, estPt_sp[1]), (255, 0, 0), 4)
        cv2.line(image1, (estPt_sp[0], estPt_sp[1] - 50), (estPt_sp[0], estPt_sp[1] + 50), (255, 0, 0), 4)
    if len(estPt_gs) > 0:
        cv2.circle(image1, estPt_gs, 5, (0, 0, 255), -1)  # 绘制目标图像的红色特征点
        cv2.line(image1, (estPt_gs[0] - 25, estPt_gs[1]), (estPt_gs[0] + 25, estPt_gs[1]), (0, 0, 255), 2)
        cv2.line(image1, (estPt_gs[0], estPt_gs[1] - 25), (estPt_gs[0], estPt_gs[1] + 25), (0, 0, 255), 2)

    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0:, :] = image1
    draw = 1
    if draw:
        # draw matched points
        mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
        # mkpts0 = [mkpts0[x] for x in mPtsIdxInRect]
        # mkpts1 = [mkpts1[x] for x in mPtsIdxInRect]
        color = (np.array(ptscolor[:, :3]) * 255).astype(int)[:, ::-1]
        for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
            c = c.tolist()
            cv2.line(out, (x0, y0), (x1 + W0, y1),
                     [255, 0, 0], thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, [0, 0, 255], -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, [0, 0, 255], -1,
                       lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 0)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (660, Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (660, Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)
    return ax


def plot_keypoints(kpts, colors='lime', ps=4, alpha=1):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c in zip(axes, kpts, colors):
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, alpha=alpha, linewidths=0)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=color[i], linewidth=lw,
            alpha=a)
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_lines(lines, line_colors='orange', point_colors='cyan',
               ps=4, lw=2, alpha=1., indices=(0, 1)):
    """ Plot lines and endpoints for existing images.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float pixels.
        lw: line width as float pixels.
        alpha: transparency of the points and lines.
        indices: indices of the images to draw the matches on.
    """
    if not isinstance(line_colors, list):
        line_colors = [line_colors] * len(lines)
    if not isinstance(point_colors, list):
        point_colors = [point_colors] * len(lines)

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines and junctions
    for a, l, lc, pc in zip(axes, lines, line_colors, point_colors):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D((l[i, 0, 0], l[i, 1, 0]),
                                           (l[i, 0, 1], l[i, 1, 1]),
                                           zorder=1, c=lc, linewidth=lw,
                                           alpha=alpha)
            a.add_line(line)
        pts = l.reshape(-1, 2)
        a.scatter(pts[:, 0], pts[:, 1],
                  c=pc, s=ps, linewidths=0, zorder=2, alpha=alpha)


def plot_color_line_matches(lines, correct_matches=None,
                            lw=2, indices=(0, 1)):
    """Plot line matches for existing images with multiple colors.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        correct_matches: bool array of size (N,) indicating correct matches.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    n_lines = len(lines[0])
    colors = sns.color_palette('husl', n_colors=n_lines)
    np.random.shuffle(colors)
    alphas = np.ones(n_lines)
    # If correct_matches is not None, display wrong matches with a low alpha
    if correct_matches is not None:
        alphas[~np.array(correct_matches)] = 0.2

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines
    for a, l in zip(axes, lines):
        # Transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        endpoint0 = transFigure.transform(a.transData.transform(l[:, 0]))
        endpoint1 = transFigure.transform(a.transData.transform(l[:, 1]))
        fig.lines += [matplotlib.lines.Line2D(
            (endpoint0[i, 0], endpoint1[i, 0]),
            (endpoint0[i, 1], endpoint1[i, 1]),
            zorder=1, transform=fig.transFigure, c=colors[i],
            alpha=alphas[i], linewidth=lw) for i in range(n_lines)]


def extract_number(filename):
    match = re.search(r'mm(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    else:
        return None


import cv2
import numpy as np

global m1br, m1tl


def warp_corners_and_draw_matches(mkpts0, mkpts1, mPointsIdx, Pt_gs, img1, img2, tempRect):

    # 使用 USAC_MAGSAC 算法计算单应性矩阵 H
    # rect_coords = np.array(tempRect, dtype=np.int32)
    # cv2.polylines(img1, [rect_coords], isClosed=True, color=(255, 0, 0), thickness=2)

    # rect_mask = np.zeros(len(mkpts0), dtype=bool)
    # for i, point in enumerate(mkpts0):
    #     x, y = point
    #     if cv2.pointPolygonTest(rect_coords, (x, y), measureDist=False) >= 0:
    #         rect_mask[i] = True

    # x1 = Pt_gs[0] - 100
    # y1 = Pt_gs[1] - 50
    # x2 = Pt_gs[0] + 100
    # y2 = Pt_gs[1] + 50
    # cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # rect_mask = ((mkpts0[:, 0] >= x1) & (mkpts0[:, 0] <= x2) &
    #              (mkpts0[:, 1] >= y1) & (mkpts0[:, 1] <= y2))
    # rect_ref_points = mkpts0[rect_mask]
    # rect_dst_points = mkpts1[rect_mask]
    #
    # H2, mask = cv2.findHomography(rect_ref_points, rect_dst_points, cv2.USAC_MAGSAC,
    #                               ransacReprojThreshold=0.1, maxIters=10000, confidence=0.9999)
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=0.1, maxIters=10000,
                                 confidence=0.9999)

    # 在本次模板 img1 上绘制 tempRect 矩形框
    # cv2.rectangle(img1, tuple(tempRect[0]), tuple(tempRect[2]), (0, 0, 255), 3)

    # 提取匹配点并绘制特征点最小外接矩形框
    if len(mPointsIdx) >= 5:
        mk0 = [mkpts0[x] for x in mPointsIdx]
        mk1 = [mkpts1[x] for x in mPointsIdx]

        # 计算 mk0 的矩形框
        x0 = [int(point[0]) for point in mk0]
        y0 = [int(point[1]) for point in mk0]
        m0tl = (min(x0), min(y0))
        m0br = (max(x0), max(y0))

        # 计算 mk1 的矩形框
        x1 = [int(point[0]) for point in mk1]
        y1 = [int(point[1]) for point in mk1]
        m1tl = (min(x1), min(y1))
        m1br = (max(x1), max(y1))

        # cv2.rectangle(img1, m0tl, m0br, (255, 0, 0), 3)
        cv2.rectangle(img2, m1tl, m1br, (0, 0, 255), 3)

    # 绘制交叉线和交点
    cross_size = 25
    cross_thickness = 2
    cross_color_1 = (255, 0, 0)
    cross_color_2 = (0, 0, 255)

    pt_a = Pt_gs
    pt_a = tuple(np.squeeze(np.int32(pt_a)))

    # 在 img1 上绘制交叉线和交点
    # cv2.line(img1, (pt_a[0] - cross_size, pt_a[1]), (pt_a[0] + cross_size, pt_a[1]), cross_color_1, cross_thickness)
    # cv2.line(img1, (pt_a[0], pt_a[1] - cross_size), (pt_a[0], pt_a[1] + cross_size), cross_color_1, cross_thickness)
    # cv2.putText(img1, f'Pt_gs: ({pt_a[0]}, {pt_a[1]})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    #             cv2.LINE_AA)

    Pt_gs = np.array([Pt_gs], dtype=np.float32).reshape((1, 1, 2))

    # 计算透视变换后的点 pt_xf
    pt_xf = cv2.perspectiveTransform(Pt_gs, H)
    pt_b = pt_xf

    pt_xf = tuple(np.squeeze(np.int32(pt_xf)))

    # 在 img2 上绘制透视变换后的点的交叉线和交点
    cv2.line(img2, (pt_xf[0] - cross_size, pt_xf[1]), (pt_xf[0] + cross_size, pt_xf[1]), cross_color_2, cross_thickness)
    cv2.line(img2, (pt_xf[0], pt_xf[1] - cross_size), (pt_xf[0], pt_xf[1] + cross_size), cross_color_2, cross_thickness)
    cv2.putText(img2, f'pt_xf: ({pt_xf[0]}, {pt_xf[1]})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)

    img2_with_corners = img2.copy()

    img = cv2.imread('E:\\Image-matching-model\\600.jpg')
    ptos = [263, 361]
    cv2.line(img, (ptos[0] - 25, ptos[1]), (ptos[0] + 25, ptos[1]), (0, 255, 0), 2)
    cv2.line(img, (ptos[0], ptos[1] - 25), (ptos[0], ptos[1] + 25), (0, 255, 0), 2)

    # 创建一个用于显示的合成图像 img_matches
    h1, w1 = img.shape[:2]
    h2, w2 = img2.shape[:2]
    img_matches = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
    img_matches[:, :w1] = img
    img_matches[:, w1:] = img2_with_corners

    return img_matches, pt_b, m1tl, m1br


def MatchesInObjRect(points, rect):
    # 检查 rect 的格式和内容
    if len(rect) != 4:
        raise ValueError("rect should contain exactly 4 points")

    # 创建 Polygon 对象
    rectangle = Polygon(rect)
    result = []
    for i in range(len(points)):
        point = Point(points[i][0], points[i][1])
        if rectangle.contains(point):
            result.append(i)
    return result


def warp_corners_and_draw_matches2(mkpts0, mkpts1, mPointsIdx, Pt_gs, img1, img2, tempRect, q, p, idx):
    a1 = Pt_gs
    a1 = tuple(np.squeeze(np.int32(a1)))
    x1 = a1[0] - q
    y1 = a1[1] - p
    x2 = a1[0] + q
    y2 = a1[1] + p
    cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
    rect_mask = ((mkpts0[:, 0] >= x1) & (mkpts0[:, 0] <= x2) &
                 (mkpts0[:, 1] >= y1) & (mkpts0[:, 1] <= y2))
    rect_ref_points = mkpts0[rect_mask]
    rect_dst_points = mkpts1[rect_mask]

    H2, mask = cv2.findHomography(rect_ref_points, rect_dst_points, cv2.USAC_MAGSAC,
                                  ransacReprojThreshold=0.1, maxIters=10000, confidence=0.9999)
    # H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=0.1, maxIters=10000,
    #                              confidence=0.9999)

    # 在本次模板 img1 上绘制 tempRect 矩形框
    # cv2.rectangle(img1, tuple(tempRect[0]), tuple(tempRect[2]), (0, 0, 255), 3)

    # 提取匹配点并绘制特征点最小外接矩形框
    if len(mPointsIdx) >= 5:
        mk1 = [mkpts1[x] for x in mPointsIdx]

        # 计算 mk1 的矩形框
        x1 = [int(point[0]) for point in mk1]
        y1 = [int(point[1]) for point in mk1]
        m1tl = (min(x1), min(y1))
        m1br = (max(x1), max(y1))

        # cv2.rectangle(img1, m0tl, m0br, (255, 0, 0), 3)
        cv2.rectangle(img2, m1tl, m1br, (0, 0, 255), 3)

    # 绘制交叉线和交点
    cross_size = 25
    cross_thickness = 2
    cross_color_1 = (255, 0, 0)
    cross_color_2 = (0, 0, 255)

    pt_a = Pt_gs
    pt_a = tuple(np.squeeze(np.int32(pt_a)))

    # 在 img1 上绘制交叉线和交点
    # cv2.line(img1, (pt_a[0] - cross_size, pt_a[1]), (pt_a[0] + cross_size, pt_a[1]), cross_color_1, cross_thickness)
    # cv2.line(img1, (pt_a[0], pt_a[1] - cross_size), (pt_a[0], pt_a[1] + cross_size), cross_color_1, cross_thickness)
    # cv2.putText(img1, f'Pt_gs: ({pt_a[0]}, {pt_a[1]})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    #             cv2.LINE_AA)

    Pt_gs = np.array([Pt_gs], dtype=np.float32).reshape((1, 1, 2))

    # 计算透视变换后的点 pt_xf
    pt_xf = cv2.perspectiveTransform(Pt_gs, H2)
    pt_b = pt_xf

    pt_xf = tuple(np.squeeze(np.int32(pt_xf)))

    # 在 img2 上绘制透视变换后的点的交叉线和交点
    cv2.line(img2, (pt_xf[0] - cross_size, pt_xf[1]), (pt_xf[0] + cross_size, pt_xf[1]), cross_color_2, cross_thickness)
    cv2.line(img2, (pt_xf[0], pt_xf[1] - cross_size), (pt_xf[0], pt_xf[1] + cross_size), cross_color_2, cross_thickness)
    cv2.putText(img2, f'pt_xf: ({pt_xf[0]}, {pt_xf[1]})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)

    img2_with_corners = img2.copy()

    img = cv2.imread('E:\\Image-matching-model\\600.jpg')
    ptos = [263, 361]
    cv2.line(img, (ptos[0] - 25, ptos[1]), (ptos[0] + 25, ptos[1]), (0, 255, 0), 2)
    cv2.line(img, (ptos[0], ptos[1] - 25), (ptos[0], ptos[1] + 25), (0, 255, 0), 2)

    # 创建一个用于显示的合成图像 img_matches
    h1, w1 = img.shape[:2]
    h2, w2 = img2.shape[:2]
    img_matches = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
    img_matches[:, :w1] = img
    img_matches[:, w1:] = img2_with_corners

    return img_matches, pt_b, m1tl, m1br
