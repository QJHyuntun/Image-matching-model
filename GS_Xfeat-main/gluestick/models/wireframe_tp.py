import numpy as np
import torch
from pytlsd import lsd

from torch import nn

from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from superpoint_tp import SuperPoint, sample_descriptors
from utils import scatter_reduce_mean

# @torch.jit.script_if_tracing
# class DBSCAN(nn.Module):
#     def __init__(self, eps=0.5, min_samples=5):
#         super().__init__()
#         self.eps = eps
#         self.min_samples = min_samples
#
#     def fit(self, X):
#         # 标记所有点为未访问
#         visited = torch.zeros(X.shape[0], dtype=torch.bool)
#         # 初始化所有点的簇标签为-1，即噪声点
#         labels = torch.full((X.shape[0],), -1, dtype=torch.int)
#         cluster_id = 0
#
#         # 对每个点进行处理
#         for idx in range(X.shape[0]):
#             if visited[idx]:
#                 continue
#             visited[idx] = True
#
#             # 找到核心点的所有直接密度可达点
#             neighbors = self._region_query(X, idx)
#
#             # 如果邻居少于min_samples，标记为噪声
#             if neighbors.shape[0] < self.min_samples:
#                 labels[idx] = -1
#             else:
#                 # 如果是核心点，创建新的簇
#                 labels[idx] = cluster_id
#                 self._expand_cluster(X, labels, visited, neighbors, cluster_id)
#                 cluster_id += 1
#
#         self.labels_ = labels
#         return self
#
#     def _expand_cluster(self, X, labels, visited, neighbors, cluster_id):
#         # 遍历核心点的所有邻居
#         i = 0
#         while i < neighbors.size(0):
#             neighbor_idx = neighbors[i].long()
#             if not visited[neighbor_idx]:
#                 visited[neighbor_idx] = True
#                 new_neighbors = self._region_query(X, neighbor_idx)
#                 # 如果邻居是核心点，添加其邻居到待处理列表
#                 if new_neighbors.size(0) >= self.min_samples:
#                     neighbors = torch.cat((neighbors, new_neighbors))
#
#             # 如果邻居点还没有被任何簇包含，将其添加到当前簇
#             if labels[neighbor_idx] == -1:
#                 labels[neighbor_idx] = cluster_id
#             i += 1
#
#     def _region_query(self, X, idx):
#         # 计算点idx与所有点之间的距离
#         point_dist = torch.norm(X - X[idx], dim=1)
#         # 获取距离小于eps的点的索引
#         neighbors = torch.nonzero(point_dist <= self.eps, as_tuple=False).view(-1)
#         return neighbors

@torch.jit.script_if_tracing
class DBSCAN(nn.Module):
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        visited = torch.zeros(X.shape[0], dtype=torch.bool)
        labels = torch.full((X.shape[0],), -1, dtype=torch.int64)
        cluster_id = 0

        for idx in range(X.shape[0]):
            if visited[idx]:
                continue
            visited[idx] = True
            neighbors = self._region_query(X, idx)
            if len(neighbors) < self.min_samples:
                labels[idx] = -1  # Mark as noise
            else:
                labels = self._expand_cluster(X, labels, visited, neighbors, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        return self

    def _expand_cluster(self, X, labels, visited, neighbors, cluster_id):
        neighbors_set = set(neighbors.tolist())
        while neighbors_set:
            neighbor_idx = neighbors_set.pop()
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors_set.update(new_neighbors.tolist())
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
        return labels

    def _region_query(self, X, idx):
        point_dist = torch.norm(X - X[idx], dim=1)
        neighbors = torch.where(point_dist <= self.eps)[0]
        return neighbors

class SPWireframeDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.superpoint_nms_radius = 4
        self.nms_radius = 3.

        self.sp = SuperPoint()

        self.dbscan = DBSCAN(eps=self.nms_radius, min_samples=1)

    def forward(self, data, lines, line_scores):
        # b_size, c, h, w = data.shape
        b_size = data.shape[0]
        h = data.shape[2]
        w = data.shape[3]
        device = data.device

        line_scores /= (torch.tensor(1e-8, dtype=line_scores.dtype, device=line_scores.device) + line_scores.max(
            dim=1).values[:, None])

        # SuperPoint prediction
        keypoints, keypoint_scores, descriptors, all_descriptors = self.sp(data)

        # Remove keypoints that are too close to line endpoints
        kp = keypoints
        line_endpts = lines.reshape(b_size, -1, 2)
        dist_pt_lines = torch.norm(
            kp[:, :, None] - line_endpts[:, None], dim=-1)
        # For each keypoint, mark it as valid or to remove
        pts_to_remove = torch.any(
            dist_pt_lines < self.superpoint_nms_radius, dim=2)
        # Simply remove them (we assume batch_size = 1 here)
        # assert len(kp) == 1
        # assert kp.shape[0] == 1
        keypoints = keypoints[0][~pts_to_remove[0]][None]
        keypoint_scores = keypoint_scores[0][~pts_to_remove[0]][None]
        descriptors = descriptors[0].T[~pts_to_remove[0]].T[None]

        # Connect the lines together to form a wireframe
        orig_lines = lines.clone()

        # Merge first close-by endpoints to connect lines
        (line_points, line_pts_scores, line_descs, line_association,
         lines, lines_junc_idx, num_true_junctions) = self.lines_to_wireframe(
            lines, line_scores, all_descriptors,
            nms_radius=self.nms_radius)

        # Add the keypoints to the junctions and fill the rest with random keypoints
        (all_points, all_scores, all_descs,
         pl_associativity) = [], [], [], []
        bs=0
        all_points.append(torch.cat(
            [line_points[bs], keypoints[bs]], dim=0))
        all_scores.append(torch.cat(
            [line_pts_scores[bs], keypoint_scores[bs]], dim=0))
        all_descs.append(torch.cat(
            [line_descs[bs], descriptors[bs]], dim=1))

        all_points = torch.stack(all_points, dim=0)
        all_scores = torch.stack(all_scores, dim=0)
        all_descs = torch.stack(all_descs, dim=0)
        # pl_associativity = torch.stack(pl_associativity, dim=0)

        del all_descriptors  # Remove dense descriptors to save memory
        torch.cuda.empty_cache()

        return all_points, all_scores, all_descs, lines, lines_junc_idx, line_scores

    def lines_to_wireframe(self, lines: torch.Tensor, line_scores: torch.Tensor, all_descs: torch.Tensor,
                           nms_radius: torch.Tensor) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[
            torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        b_size, _, _, _ = all_descs.shape
        device = lines.device
        endpoints = lines.reshape(b_size, -1, 2)

        junctions, junc_scores, junc_descs, connectivity, new_lines, lines_junc_idx, num_true_junctions = [], [], [], [], [], [], []
        for bs in range(b_size):
            self.dbscan.fit(endpoints[bs])
            clusters = self.dbscan.labels_
            # n_clusters = len(set(clusters))
            unique_elements = torch.unique(clusters)  # 本句和后一句，与上一句等价，目的是兼容TorchScript
            n_clusters = unique_elements.shape[0]
            num_true_junctions.append(n_clusters)

            clusters = torch.tensor(clusters, dtype=torch.long, device=device)
            new_junc = torch.zeros(n_clusters, 2, dtype=torch.float, device=device)
            # new_junc.scatter_reduce_(0, clusters[:, None].repeat(1, 2), endpoints[bs], reduce='mean', include_self=False)
            new_junc = scatter_reduce_mean(0, clusters[:, None].repeat(1, 2), endpoints[bs], new_junc.shape)
            junctions.append(new_junc)

            new_scores = torch.zeros(n_clusters, dtype=torch.float, device=device)
            # new_scores.scatter_reduce_(0, clusters, torch.repeat_interleave(line_scores[bs], 2), reduce='mean', include_self=False)
            new_scores = scatter_reduce_mean(0, clusters, torch.repeat_interleave(line_scores[bs], 2), new_scores.shape)
            junc_scores.append(new_scores)

            new_lines.append(junctions[-1][clusters].reshape(-1, 2, 2))
            lines_junc_idx.append(clusters.reshape(-1, 2))

            junc_connect = torch.eye(n_clusters, dtype=torch.bool, device=device)
            pairs = clusters.reshape(-1, 2)
            junc_connect[pairs[:, 0], pairs[:, 1]] = True
            junc_connect[pairs[:, 1], pairs[:, 0]] = True
            connectivity.append(junc_connect)

            junc_descs.append(sample_descriptors(junctions[-1][None], all_descs[bs:(bs + 1)], 8)[0])

        new_lines = torch.stack(new_lines, dim=0)
        lines_junc_idx = torch.stack(lines_junc_idx, dim=0)
        return (junctions, junc_scores, junc_descs, connectivity, new_lines, lines_junc_idx, num_true_junctions)