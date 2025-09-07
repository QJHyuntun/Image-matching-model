"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

import numpy as np
import torch
from torch import nn

from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from wireframe_tp import SPWireframeDescriptor
from gluestick_tp import GlueStick

def CheckLines(lines, scores):
    step1 = torch.any(lines != 0, dim=3)
    mask = torch.any(step1 != 0, dim=2)
    non_zero_lines = lines[:, mask[0]]
    non_zero_scores = scores[:, mask[0]]
    return non_zero_lines, non_zero_scores/100.

class TwoViewPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = SPWireframeDescriptor()
        self.matcher = GlueStick()

    def forward(self, img0, l0, s0, img1, l1, s1):
        l0_, s0_ = CheckLines(l0, s0)
        l1_, s1_ = CheckLines(l1, s1)
        (kpts0, kpts0_scores, kpts0_descs, lines0, lines0_junc_idx, lines0_scores) = self.extractor(img0, l0_, s0_)
        (kpts1, kpts1_scores, kpts1_descs, lines1, lines1_junc_idx, lines1_scores) = self.extractor(img1, l1_, s1_)

        m0, m1, mscores0_kpts, mscores1_kpts, m0_lines, m1_lines, mscores0_lines, mscores1_lines = \
            self.matcher(torch.Size(img0.shape), torch.Size(img1.shape),
                         kpts0, kpts0_scores, kpts0_descs, lines0, lines0_junc_idx, lines0_scores,
                         kpts1, kpts1_scores, kpts1_descs, lines1, lines1_junc_idx, lines1_scores)


        return kpts0, kpts1, m0, mscores0_kpts, lines0, lines1, m0_lines, mscores0_lines
