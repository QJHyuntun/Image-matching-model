import os.path
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.utils.checkpoint
from torch import nn

from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ETH_EPS = 1e-8

from utils import scatter_reduce_mean, scatter_reduce_sum

class GlueStick(nn.Module):
    def __init__(self):
        super().__init__()

        self.descriptor_dim = 256
        self.keypoint_encoder = [32, 64, 128, 256]
        self.GNN_layers = ['self', 'cross'] * 9
        self.num_line_iterations = 1
        self.filter_threshold = 0.2

        self.kenc = KeypointEncoder(self.descriptor_dim, self.keypoint_encoder)
        self.lenc = EndPtEncoder(self.descriptor_dim, self.keypoint_encoder)
        self.gnn = AttentionalGNN(self.descriptor_dim, self.GNN_layers,
                                  checkpointed=False,
                                  inter_supervision=None,
                                  num_line_iterations=self.num_line_iterations)
        self.final_proj = nn.Conv1d(self.descriptor_dim, self.descriptor_dim,
                                    kernel_size=1)
        nn.init.constant_(self.final_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_proj.weight, gain=1)
        self.final_line_proj = nn.Conv1d(
            self.descriptor_dim, self.descriptor_dim, kernel_size=1)
        nn.init.constant_(self.final_line_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_line_proj.weight, gain=1)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        line_bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('line_bin_score', line_bin_score)

        FILE = Path(__file__).resolve()
        ROOT = FILE.parent.parent  # root directory
        weights = ROOT / 'weights' / 'checkpoint_GlueStick_MD.tar'
        state_dict = torch.load(weights, map_location='cpu')

        if 'model' in state_dict:
            state_dict = {k.replace('matcher.', ''): v for k, v in state_dict['model'].items() if 'matcher.' in k}
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)

    def forward(self, image_size0, image_size1, \
                kpts0, keypoint_scores0, desc0, lines0, lines_junc_idx0, line_scores0, \
                kpts1, keypoint_scores1, desc1, lines1, lines_junc_idx1, line_scores1):
        device = kpts0.device
        # b_size = len(kpts0)
        b_size = kpts0.shape[0]

        n_kpts0, n_kpts1 = kpts0.shape[1], kpts1.shape[1]
        n_lines0, n_lines1 = lines0.shape[1], lines1.shape[1]

        lines0 = lines0.flatten(1, 2)
        lines1 = lines1.flatten(1, 2)
        lines_junc_idx0 = lines_junc_idx0.flatten(1, 2)  # [b_size, num_lines * 2]
        lines_junc_idx1 = lines_junc_idx1.flatten(1, 2)

        kpts0 = normalize_keypoints(kpts0, image_size0)
        kpts1 = normalize_keypoints(kpts1, image_size1)

        desc0 = desc0 + self.kenc(kpts0, keypoint_scores0)
        desc1 = desc1 + self.kenc(kpts1, keypoint_scores1)

        # Pre-compute the line encodings
        lines0 = normalize_keypoints(lines0, image_size0).reshape(
            b_size, n_lines0, 2, 2)
        lines1 = normalize_keypoints(lines1, image_size1).reshape(
            b_size, n_lines1, 2, 2)
        line_enc0 = self.lenc(lines0, line_scores0)
        line_enc1 = self.lenc(lines1, line_scores1)

        desc0, desc1 = self.gnn(desc0, desc1, line_enc0, line_enc1,
                                lines_junc_idx0, lines_junc_idx1)

        # Match all points (KP and line junctions)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        kp_scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        kp_scores = kp_scores / self.descriptor_dim ** .5
        kp_scores = log_double_softmax(kp_scores, self.bin_score)
        m0, m1, mscores0, mscores1 = self._get_matches(kp_scores)

        # Match the lines
        # d0 = desc0[:, :, :2 * n_lines0]
        # d1 = desc1[:, :, :2 * n_lines1]
        indices0 = torch.arange(0, 2 * n_lines0, dtype=torch.long, device=desc0.device)
        d0 = torch.index_select(desc0, 2, indices0)
        indices1 = torch.arange(0, 2 * n_lines1, dtype=torch.long, device=desc0.device)
        d1 = torch.index_select(desc1, 2, indices1)
        (line_scores, m0_lines, m1_lines, mscores0_lines,
         mscores1_lines, raw_line_scores) = self._get_line_matches(d0, d1,
            lines_junc_idx0, lines_junc_idx1, self.final_line_proj)
        # (line_scores, m0_lines, m1_lines, mscores0_lines,
        #  mscores1_lines, raw_line_scores) = self._get_line_matches(
        #     desc0[:, :, :2 * n_lines0], desc1[:, :, :2 * n_lines1],
        #     lines_junc_idx0, lines_junc_idx1, self.final_line_proj)

        return m0, m1, mscores0, mscores1, m0_lines, m1_lines, mscores0_lines, mscores1_lines

    def _get_matches(self, scores_mat):
        max0 = scores_mat[:, :-1, :-1].max(2)
        max1 = scores_mat[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices

        # mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        # mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        arange_like_m0 = arange_like(m0, 1)
        arange_like_m0_expanded = torch.zeros(1, arange_like_m0.size(0), device=m0.device)  # 创建一个形状为(1, N)的张量
        arange_like_m0_expanded[0] = arange_like_m0
        mutual0 = torch.eq(arange_like_m0_expanded, m1.gather(1, m0))
        arange_like_m1 = arange_like(m1, 1)
        arange_like_m1_expanded = torch.zeros(1, arange_like_m1.size(0), device=m1.device)  # 创建一个形状为(1, N)的张量
        arange_like_m1_expanded[0] = arange_like_m1
        mutual1 = torch.eq(arange_like_m1_expanded, m0.gather(1, m1))

        zero = scores_mat.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)

        valid0 = mutual0 & (mscores0 > self.filter_threshold)
        # gt_filter = torch.gt(mscores0, self.filter_threshold)
        # valid0 = torch.logical_and(mutual0, gt_filter)

        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))
        return m0, m1, mscores0, mscores1

    def _get_line_matches(self, ldesc0, ldesc1, lines_junc_idx0,
                          lines_junc_idx1, final_proj):
        mldesc0 = final_proj(ldesc0)
        mldesc1 = final_proj(ldesc1)

        line_scores = torch.einsum('bdn,bdm->bnm', mldesc0, mldesc1)
        line_scores = line_scores / self.descriptor_dim ** .5

        # Get the line representation from the junction descriptors
        n2_lines0 = lines_junc_idx0.shape[1]
        n2_lines1 = lines_junc_idx1.shape[1]
        line_scores = torch.gather(
            line_scores, dim=2,
            index=lines_junc_idx1[:, None, :].repeat(1, line_scores.shape[1], 1))
        line_scores = torch.gather(
            line_scores, dim=1,
            index=lines_junc_idx0[:, :, None].repeat(1, 1, n2_lines1))
        line_scores = line_scores.reshape((-1, n2_lines0 // 2, 2,
                                           n2_lines1 // 2, 2))

        # Match either in one direction or the other
        raw_line_scores = 0.5 * torch.maximum(
            line_scores[:, :, 0, :, 0] + line_scores[:, :, 1, :, 1],
            line_scores[:, :, 0, :, 1] + line_scores[:, :, 1, :, 0])
        line_scores = log_double_softmax(raw_line_scores, self.line_bin_score)
        m0_lines, m1_lines, mscores0_lines, mscores1_lines = self._get_matches(
            line_scores)
        return (line_scores, m0_lines, m1_lines, mscores0_lines,
                mscores1_lines, raw_line_scores)

def MLP(channels, do_bn=True):
    # n = len(channels)
    n = torch.tensor(channels).size(0)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            # if do_bn:
            #     layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def normalize_keypoints(kpts, shape_or_size):
    is_shape = isinstance(shape_or_size, (tuple, list, torch.Size))
    size_shape = torch.tensor([shape_or_size[-1], shape_or_size[-2]], dtype=torch.float32, device=kpts.device)
    size_tensor = torch.tensor(1, dtype=torch.float32, device=kpts.device)
    size = size_shape.unsqueeze(0) * is_shape + size_tensor * (not is_shape)
    c = size / 2
    f = size.max(1, keepdim=True).values * 0.7

    c = c.unsqueeze(1)
    f = f.unsqueeze(1)
    # return (kpts - c[:, None, :]) / f[:, None, :]
    c = c.expand(*kpts.shape)
    f = f.expand(*kpts.shape)
    return (kpts - c) / f

class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


class EndPtEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([5] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, endpoints, scores):
        # endpoints should be [B, N, 2, 2]
        # output is [B, feature_dim, N * 2]
        b_size, n_pts, _, _ = endpoints.shape
        # assert tuple(endpoints.shape[-2:]) == (2, 2)
        endpt_offset = (endpoints[:, :, 1] - endpoints[:, :, 0]).unsqueeze(2)
        endpt_offset = torch.cat([endpt_offset, -endpt_offset], dim=2)
        endpt_offset = endpt_offset.reshape(b_size, 2 * n_pts, 2).transpose(1, 2)
        inputs = [endpoints.flatten(1, 2).transpose(1, 2),
                  endpt_offset, scores.repeat(1, 2).unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.prob = []

    def forward(self, query, key, value):
        b = query.size(0)

        # query, key, value = [l(x).view(b, self.dim, self.h, -1)
        #                      for l, x in zip(self.proj, (query, key, value))]
        query_projected = self.proj[0](query).view(b, self.dim, self.h, -1)
        key_projected = self.proj[1](key).view(b, self.dim, self.h, -1)
        value_projected = self.proj[2](value).view(b, self.dim, self.h, -1)
        query, key, value = (query_projected, key_projected, value_projected)

        x, prob = attention(query, key, value)
        # self.prob.append(prob.mean(dim=1))
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads, skip_init=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim], do_bn=True)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        self.scaling = 1.

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)) * self.scaling


class GNNLayer(nn.Module):
    def __init__(self, feature_dim, layer_type):
        super().__init__()
        assert layer_type in ['cross', 'self']
        self.type = layer_type
        self.update = AttentionalPropagation(feature_dim, 4)

    def forward(self, desc0, desc1):
        if self.type == 'cross':
            src0, src1 = desc1, desc0
        elif self.type == 'self':
            src0, src1 = desc0, desc1
        else:
            raise ValueError("Unknown layer type: " + self.type)
        # self.update.attn.prob = []
        delta0, delta1 = self.update(desc0, src0), self.update(desc1, src1)
        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class LineLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.dim = feature_dim
        self.mlp = MLP([self.dim * 3, self.dim * 2, self.dim], do_bn=True)


    def get_endpoint_update(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]
        # and lines_junc_idx [bs, n_lines * 2]
        # Create one message per line endpoint
        b_size = lines_junc_idx.shape[0]
        line_desc = torch.gather(
            ldesc, 2, lines_junc_idx[:, None].repeat(1, self.dim, 1))
        message = torch.cat([
            line_desc,
            line_desc.reshape(b_size, self.dim, -1, 2).flip([-1]).flatten(2, 3).clone(),
            line_enc], dim=1)
        return self.mlp(message)  # [b_size, D, n_lines * 2]

    def get_endpoint_attention(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]
        # and lines_junc_idx [bs, n_lines * 2]
        b_size = lines_junc_idx.shape[0]
        expanded_lines_junc_idx = lines_junc_idx[:, None].repeat(1, self.dim, 1)

        # Query: desc of the current node
        query = self.proj_node(ldesc)  # [b_size, D, n_junc]
        query = torch.gather(query, 2, expanded_lines_junc_idx)
        # query is [b_size, D, n_lines * 2]

        # Key: combination of neighboring desc and line encodings
        line_desc = torch.gather(ldesc, 2, expanded_lines_junc_idx)
        key = self.proj_neigh(torch.cat([
            line_desc.reshape(b_size, self.dim, -1, 2).flip([-1]).flatten(2, 3).clone(),
            line_enc], dim=1))  # [b_size, D, n_lines * 2]

        # Compute the attention weights with a custom softmax per junction
        prob = (query * key).sum(dim=1) / self.dim ** .5  # [b_size, n_lines * 2]
        prob = torch.exp(prob - prob.max())
        # denom = torch.zeros_like(ldesc[:, 0]).scatter_reduce_(
        #     dim=1, index=lines_junc_idx,
        #     src=prob, reduce='sum', include_self=False)  # [b_size, n_junc]
        denom = scatter_reduce_sum(ldesc, lines_junc_idx, prob)  # [b_size, n_junc]

        denom = torch.gather(denom, 1, lines_junc_idx)  # [b_size, n_lines * 2]
        prob = prob / (denom + ETH_EPS)
        return prob  # [b_size, n_lines * 2]

    def forward(self, ldesc0, ldesc1, line_enc0, line_enc1, lines_junc_idx0,
                lines_junc_idx1):
        # Gather the endpoint updates
        lupdate0 = self.get_endpoint_update(ldesc0, line_enc0, lines_junc_idx0)
        lupdate1 = self.get_endpoint_update(ldesc1, line_enc1, lines_junc_idx1)

        update0, update1 = torch.zeros_like(ldesc0), torch.zeros_like(ldesc1)
        dim = ldesc0.shape[1]

        # Average the updates for each junction (requires torch > 1.12)
        # update0 = update0.scatter_reduce_(
        #     dim=2, index=lines_junc_idx0[:, None].repeat(1, dim, 1),
        #     src=lupdate0, reduce='mean', include_self=False)
        # update1 = update1.scatter_reduce_(
        #     dim=2, index=lines_junc_idx1[:, None].repeat(1, dim, 1),
        #     src=lupdate1, reduce='mean', include_self=False)
        update0 = scatter_reduce_mean(
            dim=2, index=lines_junc_idx0[:, None].repeat(1, dim, 1),
            src=lupdate0, out_shape=ldesc0.shape)
        update1 = scatter_reduce_mean(
            dim=2, index=lines_junc_idx1[:, None].repeat(1, dim, 1),
            src=lupdate1, out_shape=ldesc1.shape)


        # Update
        ldesc0 = ldesc0 + update0
        ldesc1 = ldesc1 + update1

        return ldesc0, ldesc1


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_types, checkpointed=False,
                 skip=False, inter_supervision=None, num_line_iterations=1):
        super().__init__()
        self.checkpointed = checkpointed
        self.inter_supervision = inter_supervision
        self.num_line_iterations = num_line_iterations
        self.layers = nn.ModuleList([
            GNNLayer(feature_dim, layer_type)
            for layer_type in layer_types])
        # self.line_layers = nn.ModuleList(
        #     [LineLayer(feature_dim)
        #      for _ in range(len(layer_types) // 2)])
        self.line_layers = nn.ModuleList(
            [LineLayer(feature_dim)
             for _ in range(9)])

    def forward(self, desc0, desc1, line_enc0, line_enc1,
                lines_junc_idx0, lines_junc_idx1):
        # for i, layer in enumerate(self.layers):
        #     desc0, desc1 = layer(desc0, desc1)
        #     # Add line self attention layers after every self layer
        #     for _ in range(self.num_line_iterations):
        #         layer_index = int(i // 2)
        #         desc0, desc1 = self.line_layers[layer_index](
        #             desc0, desc1, line_enc0, line_enc1,
        #             lines_junc_idx0, lines_junc_idx1)
        # 第一对GNN层（self和cross）以及对应的LineLayer处理
        desc0, desc1 = self.layers[0](desc0, desc1)  # 第1个self层
        desc0, desc1 = self.line_layers[0](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        desc0, desc1 = self.layers[1](desc0, desc1)  # 第1个cross层

        # 第二对GNN层（self和cross）以及对应的LineLayer处理
        desc0, desc1 = self.layers[2](desc0, desc1)  # 第2个self层
        desc0, desc1 = self.line_layers[1](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        desc0, desc1 = self.layers[3](desc0, desc1)  # 第2个cross层

        # 第三对GNN层
        desc0, desc1 = self.layers[4](desc0, desc1)
        desc0, desc1 = self.line_layers[2](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        desc0, desc1 = self.layers[5](desc0, desc1)

        # 第四对GNN层
        desc0, desc1 = self.layers[6](desc0, desc1)
        desc0, desc1 = self.line_layers[3](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        desc0, desc1 = self.layers[7](desc0, desc1)

        # 第五对GNN层
        desc0, desc1 = self.layers[8](desc0, desc1)
        desc0, desc1 = self.line_layers[4](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        desc0, desc1 = self.layers[9](desc0, desc1)

        # 第六对GNN层
        desc0, desc1 = self.layers[10](desc0, desc1)
        desc0, desc1 = self.line_layers[5](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        desc0, desc1 = self.layers[11](desc0, desc1)

        # 第七对GNN层
        desc0, desc1 = self.layers[12](desc0, desc1)
        desc0, desc1 = self.line_layers[6](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        desc0, desc1 = self.layers[13](desc0, desc1)

        # 第八对GNN层
        desc0, desc1 = self.layers[14](desc0, desc1)
        desc0, desc1 = self.line_layers[7](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        desc0, desc1 = self.layers[15](desc0, desc1)

        # 第九对GNN层
        desc0, desc1 = self.layers[16](desc0, desc1)
        desc0, desc1 = self.line_layers[8](desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)
        desc0, desc1 = self.layers[17](desc0, desc1)

        return desc0, desc1

def log_double_softmax(scores, bin_score):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = bin_score.expand(b, m, 1)
    bins1 = bin_score.expand(b, 1, n)
    bin_score = bin_score.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, bin_score], -1)], 1)

    scores0 = torch.nn.functional.log_softmax(couplings, dim=1)
    scores1 = torch.nn.functional.log_softmax(couplings, dim=2)

    new_scores = (scores0 + scores1) / 2

    return new_scores

def log_double_softmax1(scores, bin_score):
    b, m, n = scores.shape
    s0_fill = torch.full((b, m, 1), bin_score.item()).to(scores.device)
    scores0 = torch.cat((scores, s0_fill), 2)
    scores0 = torch.nn.functional.log_softmax(scores0, dim=2)
    s1_fill = torch.full((b, 1, n), bin_score.item()).to(scores.device)
    scores1 = torch.cat((scores, s1_fill), 1)
    scores1 = torch.nn.functional.log_softmax(scores1, dim=1)
    new_scores = torch.full((b, m + 1, n + 1), 0)
    new_scores[:, :m, :n] = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    new_scores[:, 0:m, n] = scores0[:, :, n]
    new_scores[:, m, 0:n] = scores1[:, m, :]
    return new_scores.float()

def log_double_softmax0(scores, bin_score):
    b, m, n = scores.shape
    bin_ = bin_score[None, None, None]
    scores0 = torch.cat([scores, bin_.expand(b, m, 1)], 2)
    scores1 = torch.cat([scores, bin_.expand(b, 1, n)], 1)
    scores0 = torch.nn.functional.log_softmax(scores0, 2)
    scores1 = torch.nn.functional.log_softmax(scores1, 1)
    scores = scores.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    scores[:, 0:scores.size(1)-1, scores.size(2)-1] = scores0[:, :, scores.size(2)-1]
    scores[:, scores.size(1)-1, 0:scores.size(2)-1] = scores1[:, scores.size(1)-1, :]
    return scores

def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
