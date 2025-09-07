"""
Inference model of SuperPoint, a feature detector and descriptor.

Described in:
    SuperPoint: Self-Supervised Interest Point Detection and Description,
    Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, CVPRW 2018.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork
"""

import torch
from torch import nn

from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

def bilinear_grid_sample(input, grid, align_corners=False):
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    n, c, h, w = input.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn
    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2
    x = x.view(n, -1)
    y = y.view(n, -1)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    na = ((x1 - x) * (y1 - y)).unsqueeze(1)
    nb = ((x1 - x) * (y - y0)).unsqueeze(1)
    nc = ((x - x0) * (y1 - y)).unsqueeze(1)
    nd = ((x - x0) * (y - y0)).unsqueeze(1)
    in_padded = torch.nn.functional.pad(input, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    x0, x1, y0, y1 = (x0 + 1).to(device), (x1 + 1).to(device), (y0 + 1).to(device), (y1 + 1).to(device)
    zero = torch.tensor(0).to(device)
    w_1 = torch.tensor(padded_w - 1).to(device)
    h_1 = torch.tensor(padded_h - 1).to(device)
    x0 = torch.where(x0 < 0, zero, x0)
    x0 = torch.where(x0 > padded_w - 1, w_1, x0)
    x1 = torch.where(x1 < 0, zero, x1)
    x1 = torch.where(x1 > padded_w - 1, w_1, x1)
    y0 = torch.where(y0 < 0, zero, y0)
    y0 = torch.where(y0 > padded_h - 1, h_1, y0)
    y1 = torch.where(y1 < 0, zero, y1)
    y1 = torch.where(y1 > padded_h - 1, h_1, y1)
    in_padded = in_padded.view(n, c, -1)
    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    Ia = torch.gather(in_padded, 2, x0_y0)
    Ib = torch.gather(in_padded, 2, x0_y1)
    Ic = torch.gather(in_padded, 2, x1_y0)
    Id = torch.gather(in_padded, 2, x1_y1)

    return (Ia * na + Ib * nb + Ic * nc + Id * nd).reshape(n, c, gh, gw)

def sample_descriptors(keypoints, descriptors, s):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    # args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    # descriptors1 = torch.nn.functional.grid_sample(
    #     descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', align_corners=True)
    descriptors = bilinear_grid_sample(descriptors, keypoints.view(b, 1, -1, 2), align_corners=True)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

def simple_nms(scores, radius):
    """Perform non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Args:
        scores: the score heatmap of size `(B, H, W)`.
        size: an interger scalar, the radius of the NMS window.
    """
    # def max_pool(x):
    #     return torch.nn.functional.max_pool2d(
    #         x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_pool = nn.MaxPool2d(kernel_size=radius * 2 + 1, stride=1, padding=radius)
    max_mask = scores == max_pool(scores)
    # max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))

    return torch.where(max_mask, scores, zeros)

# class SuperPoint(BaseModel):
#     def _init(self):
class SuperPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.descriptor_dim = 256
        self.nms_radius = 4
        self.detection_threshold = 0.005
        self.max_num_keypoints = 1000
        self.remove_borders = 4

        # self.max_pool = nn.MaxPool2d(kernel_size=self.nms_radius * 2 + 1, stride=1, padding=self.nms_radius)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.descriptor_dim, kernel_size=1, stride=1, padding=0)

        FILE = Path(__file__).resolve()
        ROOT = FILE.parent.parent  # root directory
        path = ROOT / 'weights' / 'superpoint_v1.pth'
        weights = torch.load(str(path), map_location='cpu')

        self.load_state_dict(weights, strict=False)


    def forward(self, image):
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores0 = self.convPb(cPa)
        scores1 = torch.nn.functional.softmax(scores0, 1)[:, :-1]
        b, c, h, w = scores1.shape
        scores2 = scores1.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores3 = scores2.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        all_desc = self.convDb(cDa)
        all_desc = torch.nn.functional.normalize(all_desc, p=2., dim=1)

        # scores = simple_nms(scores, self.nms_radius)
        scores4 = simple_nms(scores3[None], self.nms_radius).squeeze(0)

        # Extract keypoints
        # keypoints = [torch.nonzero(s > self.detection_threshold) for s in scores]
        mask = scores4 > self.detection_threshold
        keypoints = [torch.nonzero(mask.squeeze(0))]

        # scores = [s[tuple(k.t())] for s, k in zip(scores4, keypoints)]
        scores_tensor = torch.tensor(scores4)
        keypoints_tensor = torch.stack(keypoints)
        result = []
        for i in range(keypoints_tensor.size(0)):
            k = keypoints_tensor[i]
            s = scores_tensor[i]
            index = tuple(k.t().tolist())
            result.append(s[index])
        scores = result
        # Discard keypoints near the image borders
        # keypoints, scores = list(zip(*[
        #     self.remove_borders(k, s, self.remove_borders, h * 8, w * 8)
        #     for k, s in zip(keypoints, scores)]))
        mask_h = (keypoints[0][:, 0] >= self.remove_borders) & (keypoints[0][:, 0] < (h*8 - self.remove_borders))
        mask_w = (keypoints[0][:, 1] >= self.remove_borders) & (keypoints[0][:, 1] < (w*8 - self.remove_borders))
        mask = mask_h & mask_w
        keypoints, scores = keypoints[0][mask], scores[0][mask]

        # Keep the k keypoints with highest score
        keypoints, scores = self.top_k_keypoints(keypoints, scores, self.max_num_keypoints)

        # Convert (h, w) to (x, y)
        # keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        keypoints = [keypoints.flip([1]).float()]

        # Extract descriptors
        # desc = [sample_descriptors(k[None], d[None], 8)[0]
        #         for k, d in zip(keypoints, all_desc)]
        b, c, h, w = all_desc.shape
        s=8
        kps = keypoints[0][None] - s / 2 + 0.5
        kps /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                                  ).to(kps)[None]
        kps = kps * 2 - 1  # normalize to (-1, 1)

        # descriptors = torch.nn.functional.grid_sample(all_desc, kps.view(b, 1, -1, 2), mode='bilinear', align_corners=True)
        descriptors = bilinear_grid_sample(all_desc, kps.view(b, 1, -1, 2), align_corners=True)
        desc = torch.nn.functional.normalize(descriptors.reshape(b, c, -1), p=2., dim=1)
        desc = [desc[0]]

        return keypoints[0].unsqueeze(0), scores.unsqueeze(0), desc[0].unsqueeze(0), all_desc

    # def remove_borders(self, keypoints, scores, b, h, w):
    #     mask_h = (keypoints[:, 0] >= b) & (keypoints[:, 0] < (h - b))
    #     mask_w = (keypoints[:, 1] >= b) & (keypoints[:, 1] < (w - b))
    #     mask = mask_h & mask_w
    #     return keypoints[mask], scores[mask]

    def topk(self, scores, k, dim=0, sorted=True):
        # 使用 torch.sort() 函数对分数张量进行排序
        sorted_scores, sorted_indices = torch.sort(scores, dim=0, descending=True)

        # 获取得分最高的 k 个元素的索引
        topk_indices = sorted_indices[:k]

        # 获取得分最高的 k 个元素的值
        topk_scores = scores[topk_indices]

        # 返回结果
        return topk_scores, topk_indices

    def top_k_keypoints(self, keypoints, scores, k):
        # scores, indices = torch.topk(scores, min(k, scores.numel()), dim=0, sorted=True)
        scores, indices = self.topk(scores, min(k, scores.numel()), dim=0, sorted=True)
        kpts = keypoints[indices]
        scorces_pad = nn.functional.pad(scores, (0, k - scores.numel()), value=0)
        kpts_pad = nn.functional.pad(kpts, (0, 0, 0, k-kpts.size(0)), value=0)
        return kpts_pad, scorces_pad

    def top_k_keypoints1(self, keypoints, scores, k):
        if k >= len(keypoints):
            return keypoints, scores
        scores, indices = torch.topk(scores, k, dim=0, sorted=True)
        return keypoints[indices], scores
