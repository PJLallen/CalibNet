import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

CALIBNET_DECODER_REGISTRY = Registry("CalibNet Decoder")
CALIBNET_DECODER_REGISTRY.__doc__ = "registry for CalibNet decoder"


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)

@CALIBNET_DECODER_REGISTRY.register()
class CalibNetDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # add 2 for coordinates
        input_dim = cfg.MODEL.CALIBNET.DECODER.INPUT_CHANNELS
        in_channels = cfg.MODEL.CALIBNET.DECODER.INPUT_CHANNELS + 2

        self.scale_factor = cfg.MODEL.CALIBNET.DECODER.SCALE_FACTOR

        self.inst_branch = DynamicInteractiveKernel(cfg, in_channels)

        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.rgb_input = nn.Sequential(nn.Conv2d(input_dim, 1, 1))
        self.depth_input = nn.Sequential(nn.Conv2d(input_dim, 1, 1))
        self._init_weights()

    def _init_weights(self):
        for modules in [self.rgb_input, self.depth_input]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_normal_(l.weight)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def compute_depth_similarity_assessment(self, rgb_feat, depth_feat):
        bs,c,h,w = rgb_feat.shape
        rgb_feat_cosine = self.rgb_input(rgb_feat).view(bs,-1)
        depth_feat_consine = self.depth_input(depth_feat).view(bs,-1)
        cosine_sim = self.cosine_similarity(rgb_feat_cosine, depth_feat_consine)
        cosine_sim = (1-cosine_sim) / 2.0
        cosine_sim = 1-cosine_sim
        return cosine_sim

    def forward(self, features, features_depth, mask_features):
        coord_features = self.compute_coordinates(features)

        # compute cosine simiarity
        cosine_sim = self.compute_depth_similarity_assessment(features, features_depth)
        bs = features.shape[0]
        features_depth = cosine_sim.view(bs,1,1,1) * features_depth

        features = torch.cat([features, coord_features], dim=1)
        features_depth = torch.cat([features_depth, coord_features], dim=1)
        pred_scores, pred_kernel, pred_objectness = self.inst_branch(features, features_depth)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(
            B, C, H * W)).view(B, N, H, W)

        if self.scale_factor != 1:
            pred_masks = F.interpolate(
                pred_masks, scale_factor=self.scale_factor,
                mode='bilinear', align_corners=False)

        output = {
            "pred_logits": pred_scores,
            "pred_masks": pred_masks,
            "pred_scores": pred_objectness,
        }

        return output

class DynamicInteractiveKernel(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.CALIBNET.DECODER.CONV_DIM
        num_convs = 1
        num_kernels = cfg.MODEL.CALIBNET.DECODER.NUM_KERNELS
        kernel_dim = cfg.MODEL.CALIBNET.FUSION_DECODER.MASK_DIM
        self.num_groups = 4
        self.num_classes = cfg.MODEL.CALIBNET.NUM_CLASSES

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.inst_convs_depth = _make_stack_3x3_convs(num_convs, in_channels, dim)

        expand_dim = dim * self.num_groups
        self.kernel_conv = nn.Conv2d(
            dim, num_kernels * self.num_groups, 3, padding=1, groups=self.num_groups)
        self.kernel_conv_depth = nn.Conv2d(
            dim, num_kernels * self.num_groups, 3, padding=1, groups=self.num_groups)
        # outputs
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.squeeze = nn.Linear(2*num_kernels, num_kernels)
        self.score_linear = nn.Linear(expand_dim, self.num_classes)
        self.mask_linear = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)

        self.prior_prob = 0.01

        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        for m in self.inst_convs_depth.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.kernel_conv, self.kernel_conv_depth, self.score_linear]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.kernel_conv.weight, std=0.01)
        init.normal_(self.kernel_conv_depth.weight, std=0.01)
        init.normal_(self.score_linear.weight, std=0.01)

        init.normal_(self.mask_linear.weight, std=0.01)
        init.constant_(self.mask_linear.bias, 0.0)
        c2_xavier_fill(self.fc)
        c2_xavier_fill(self.squeeze)

    def forward(self, features_rgb, features_depth):
        features_rgb = self.inst_convs(features_rgb)
        features_depth = self.inst_convs_depth(features_depth)
        kernel_feat = self.kernel_conv(features_rgb).sigmoid()
        kernel_feat_depth = self.kernel_conv_depth(features_depth).sigmoid()

        B, N = kernel_feat.shape[:2]
        C = features_rgb.size(1)
        kernel_feat = kernel_feat.view(B, N, -1)
        normalizer = kernel_feat.sum(-1).clamp(min=1e-6)
        kernel_feat = kernel_feat / normalizer[:, :, None]
        kernel_feat_depth = kernel_feat_depth.view(B, N, -1)
        normalizer = kernel_feat_depth.sum(-1).clamp(min=1e-6)
        kernel_feat_depth = kernel_feat_depth / normalizer[:, :, None]

        kernel_feat = torch.cat([kernel_feat, kernel_feat_depth], dim=1)

        inst_features = torch.bmm(
            kernel_feat, features_rgb.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(
            B, self.num_groups, 2 * N // self.num_groups, -1).transpose(1, 2).reshape(B, 2 * N // self.num_groups, -1)

        inst_features = self.squeeze(inst_features.transpose(1,2))
        inst_features = F.relu(inst_features).transpose(1,2)
        inst_features = F.relu_(self.fc(inst_features))

        pred_scores = self.score_linear(inst_features)
        pred_kernel = self.mask_linear(inst_features)
        pred_objectness = self.objectness(inst_features)
        return pred_scores, pred_kernel, pred_objectness

def build_kernel_branch(cfg):
    name = cfg.MODEL.CALIBNET.DECODER.NAME
    return CALIBNET_DECODER_REGISTRY.get(name)(cfg)