import numpy as np
import torch


# ========================================模板选择====================================================
def DistanceOfPt2Line(point, line_start, line_end):
    """计算点到直线的距离"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    projection = np.dot(point_vec, line_vec) / line_len
    if projection <= 0:
        return np.linalg.norm(point_vec)
    elif projection >= line_len:
        return np.linalg.norm(point - line_end)
    else:
        proj_vec = projection * line_vec / line_len
        return np.linalg.norm(point_vec - proj_vec)


def FindNearestLines(point, lines, max_nums=10):
    """找到离点最近的直线"""
    distances = [(index, DistanceOfPt2Line(point, line[0], line[1])) for index, line in enumerate(lines)]

    # 排序并取距离最小的前10条直线
    idx = sorted(distances, key=lambda x: x[1])[:min(len(distances), max_nums)]
    nearest_lines = [lines[index] for index, _ in idx]

    return nearest_lines


def BoundingRect(lines):
    """计算一组直线的最小外接矩形"""
    # min_x = min_y = float('inf')
    # max_x = max_y = -float('inf')
    # for line in lines:
    #     min_x = min(min_x, line[0][0], line[1][0])
    #     max_x = max(max_x, line[0][0], line[1][0])
    #     min_y = min(min_y, line[0][1], line[1][1])
    #     max_y = max(max_y, line[0][1], line[1][1])

    all_points = [point for line in lines for point in line]

    # 计算所有点的最小和最大x、y值
    min_x = min(all_points, key=lambda point: point[0])[0]
    max_x = max(all_points, key=lambda point: point[0])[0]
    min_y = min(all_points, key=lambda point: point[1])[1]
    max_y = max(all_points, key=lambda point: point[1])[1]

    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


def GenerateTemplate(pt, lines, line_num_in_rect=20):
    nearest_lines = FindNearestLines(pt, lines, line_num_in_rect)
    tempRect = BoundingRect(nearest_lines)
    return tempRect, nearest_lines


# ========================================参考点定位===================================================
from scipy.linalg import lstsq


def RefPointLocationWithLeastSquare(mKptsA, mKptsB, mLinesA, mLinesB, refPt):
    """找到由A平面上的点映射到B平面上相应点的仿射变换矩阵，并用最小二乘法估计位置"""
    A_matrix = []
    B_matrix = []
    for (ax, ay), (bx, by) in zip(mKptsA, mKptsB):
        A_matrix.append([ax, ay, 1, 0, 0, 0])
        A_matrix.append([0, 0, 0, ax, ay, 1])
        B_matrix.append(bx)
        B_matrix.append(by)

    A_matrix = np.array(A_matrix)
    B_matrix = np.array(B_matrix)

    # 解最小二乘问题找到仿射变换矩阵的参数
    params, _, _, _ = lstsq(A_matrix, B_matrix)

    # 仿射变换矩阵
    affine_matrix = params.reshape(2, 3)

    point = np.array([refPt[0], refPt[1], 1])
    estimated_point = np.dot(affine_matrix, point)

    return list(map(int, np.around(estimated_point)))


from scipy.interpolate import RBFInterpolator


def RefPointLocationWithRBF(mKptsA, mKptsB, mLinesA, mLinesB, refPt):
    # 使用RBF插值器创建非线性映射
    rbf = RBFInterpolator(mKptsA, mKptsB)

    # 估算对应B平面上的点D
    estimated_point = rbf(refPt)

    return list(map(int, np.around(estimated_point[0]).tolist()))


from sklearn.svm import SVR


def RefPointLocationWithSVR(mKptsA, mKptsB, mLinesA, mLinesB, refPt):
    # 创建支持向量机回归模型
    model = SVR(kernel='rbf')  # 使用径向基函数（RBF）作为核函数

    # 训练模型
    model.fit(mKptsA, mKptsB)

    # 使用模型预测C点在B平面上的对应点D
    estimated_point = model.predict([refPt])

    return list(map(int, np.around(estimated_point[0]).tolist()))


from sklearn.linear_model import LinearRegression


def RefPointLocationWithLinearRegression(mKptsA, mKptsB, mLinesA, mLinesB, refPt):
    # 创建线性回归模型
    model = LinearRegression()

    # 训练模型
    model.fit(mKptsA, mKptsB)

    # 使用模型预测C点在B平面上的对应点D
    estimated_point = model.predict([refPt])

    return list(map(int, np.around(estimated_point[0]).tolist()))


from sklearn.preprocessing import PolynomialFeatures


def RefPointLocationWithPolyRegression(mKptsA, mKptsB, mLinesA, mLinesB, refPt):
    # 创建多项式特征转换器
    poly_features = PolynomialFeatures(degree=2)  # 设置多项式的次数

    # 转换A平面上的点，增加多项式特征
    A_poly = poly_features.fit_transform(mKptsA)
    C_poly = poly_features.transform([refPt])

    # 创建线性回归模型
    model = LinearRegression()

    # 训练模型
    model.fit(mKptsA, mKptsB)

    # 使用模型预测C点在B平面上的对应点D
    estimated_point = model.predict([refPt])

    return list(map(int, np.around(estimated_point[0]).tolist()))


# ========================================判断点或线是否在目标矩形内部=====================================================
from shapely.geometry import Point, Polygon


def MatchedPointsInObjRect(points, rect):
    # 创建矩形对象
    rectangle = Polygon(rect)

    result = []
    for i in range(len(points)):
        point = Point(points[i][0], points[i][1])
        if rectangle.contains(point):
            result.append(i)

    return result


def MatchedLinesInObjRect(lines, rect):
    # 创建矩形对象
    rectangle = Polygon(rect)

    result = []
    for i in range(len(lines)):
        point0 = Point(lines[i][0][0], lines[i][0][1])
        point1 = Point(lines[i][1][0], lines[i][1][1])
        if rectangle.contains(point0) and rectangle.contains(point1):
            result.append(i)

    return result


import math


def IsPtInList(point, point_list, threshold=1):
    """检查一个点是否在给定的点列表内，或者与列表中的点距离小于阈值
    """
    for index, existing_point in enumerate(point_list):
        # 计算两点之间的欧几里得距离
        distance = math.sqrt((existing_point[0] - point[0]) ** 2 + (existing_point[1] - point[1]) ** 2)
        if distance < threshold:
            return index
    return None

