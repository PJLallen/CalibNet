import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

MASK_BRANCH_REGISTRY = Registry("MASK_BRANCH")
MASK_BRANCH_REGISTRY.__doc__ = "registry for mask branch"


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)

@MASK_BRANCH_REGISTRY.register()
class CalibNetFusionDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.MODEL.CALIBNET.FUSION_DECODER.INPUT_DIM
        dim = 256
        num_convs = 1
        kernel_dim = cfg.MODEL.CALIBNET.FUSION_DECODER.MASK_DIM
        self.fusion_module = WeightSharingFusion(in_channels, dim)
        self.mask_convs = _make_stack_3x3_convs(num_convs, dim+2, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

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

    def forward(self, rgb_features, depth_features):
        # compute coords
        corrd_features = self.compute_coordinates(rgb_features[2])
        # fuse rgb and depth
        feat = self.fusion_module(rgb_features[2], depth_features[2])
        feat = torch.cat([corrd_features, feat], dim=1)
        feat = self.mask_convs(feat)
        return self.projection(feat)
    
@MASK_BRANCH_REGISTRY.register()
class TwoScaleCalibNetFusionDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.MODEL.CALIBNET.FUSION_DECODER.INPUT_DIM
        dim = 256
        num_convs = 1
        kernel_dim = cfg.MODEL.CALIBNET.FUSION_DECODER.MASK_DIM
        self.fusion_module = WeightSharingFusion(in_channels, dim)
        self.fusion_module_res2 = WeightSharingFusion(in_channels, dim)
        self.bn = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()
        self.upconv = nn.ConvTranspose2d(dim+2,dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.mask_convs = _make_stack_3x3_convs(num_convs, dim, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)
        c2_msra_fill(self.upconv)

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

    def forward(self, rgb_features, depth_features):
        # compute coords
        corrd_features = self.compute_coordinates(rgb_features[2])  # t3
        # fuse rgb and depth
        feat = self.fusion_module(rgb_features[2], depth_features[2])
        feat_res2 = self.fusion_module_res2(rgb_features[3], depth_features[3])
        feat = self.relu(self.bn(feat))
        feat = torch.cat([corrd_features, feat], dim=1)
        feat = self.upconv(feat)
        feat = feat + feat_res2
        
        feat = self.mask_convs(feat)
        return self.projection(feat)


class WeightSharingFusion(nn.Module):
    def __init__(self, input_dim, conv_dim):
        super().__init__()
        self.spatial_attn_rgb = SpatialAttention(input_dim, conv_dim)
        self.spatial_attn_depth = SpatialAttention(input_dim, conv_dim)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, conv_dim),
            nn.ReLU(inplace=True)
        )
        self.smooth_conv = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, conv_dim),
            nn.ReLU(inplace=True)
        )
        self.rgb_weight_mlp = nn.Sequential(
            nn.Conv2d(1,1,1), nn.ReLU()
        )
        self.depth_weight_mlp = nn.Sequential(
            nn.Conv2d(1,1,1), nn.ReLU()
        )
        self.g1 = nn.Parameter(torch.ones(1))
        self.g2 = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.rgb_input = nn.Sequential(nn.Conv2d(input_dim, 1, 1))
        self.depth_input = nn.Sequential(nn.Conv2d(input_dim, 1, 1))

    def _init_weights(self):
        for modules in [self.depth_conv, self.smooth_conv, self.rgb_weight_mlp, self.depth_weight_mlp]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_normal_(l.weight)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    def compute_depth_similarity_assessment(self, rgb_feat, depth_feat):
        bs,c,h,w = rgb_feat.shape
        rgb_feat_cosine = self.rgb_input(rgb_feat).view(bs,-1)
        depth_feat_cosine = self.depth_input(depth_feat).view(bs,-1)
        cosine_sim = self.cosine_similarity(rgb_feat_cosine, depth_feat_cosine)
        cosine_sim = (1-cosine_sim) / 2.0
        cosine_sim = 1-cosine_sim
        return cosine_sim

    def forward(self, rgb_feat, depth_feat):
        assert(rgb_feat.shape[-1] == depth_feat.shape[-1])

        # get feat and attention map
        rgb_feat, rgb_weight = self.spatial_attn_rgb(rgb_feat)  # weight:[bs,1,h,w]
        depth_feat, depth_weight = self.spatial_attn_depth(depth_feat)
        
        # compute cosine similarity
        bs = rgb_feat.shape[0]
        cosine_sim = self.compute_depth_similarity_assessment(rgb_feat, depth_feat)
        depth_feat = cosine_sim.view(bs,1,1,1) * depth_feat

        # compute affinity
        bs,c,h,w = rgb_feat.size()
        rgb_weight = rgb_weight.squeeze(1)
        depth_weight = depth_weight.squeeze(1)
        affinity = torch.bmm(rgb_weight, depth_weight.permute(0,2,1))
        affinity_rgb = self.rgb_weight_mlp(affinity.view(bs,1,h,h))
        affinity_depth = self.depth_weight_mlp(affinity.view(bs,1,h,h))
        affinity_rgb = self.softmax(affinity_rgb.view(bs,-1)).view(bs,h,h)
        affinity_depth = self.softmax(affinity_depth.view(bs,-1)).view(bs,h,h)

        rgb_weight = (rgb_weight + self.g1 * torch.bmm(affinity_rgb, rgb_weight)).view(bs,1,h,w)
        depth_weight = (depth_weight + self.g2 * torch.bmm(affinity_depth, depth_weight)).view(bs,1,h,w)

        # spatial attention
        rgb_feat = rgb_feat * torch.sigmoid(rgb_weight)
        depth_feat = depth_feat * torch.sigmoid(depth_weight)
        depth_feat = self.depth_conv(depth_feat) 
        feat = self.smooth_conv(rgb_feat+depth_feat)
        
        return feat

class SpatialAttention(nn.Module):
    def __init__(self, input_dim, conv_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.feat_conv = nn.Sequential(
            nn.Conv2d(input_dim, conv_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
        )
        self.attn_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)

        for modules in [self.feat_conv]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_normal_(l.weight)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0) 
        nn.init.kaiming_normal_(self.attn_conv.weight)
        if self.attn_conv.bias is not None:
            nn.init.constant_(self.attn_conv.bias, 0) 

    def forward(self, feat):
        feat = self.feat_conv(feat)
        avgout = torch.mean(feat, dim=1, keepdim=True)
        maxout, _ = torch.max(feat, dim=1, keepdim=True)
        attn = torch.cat([avgout, maxout], dim=1)
        attn = self.sigmoid(self.attn_conv(attn))
        return feat, attn


def build_mask_branch(cfg):
    name = cfg.MODEL.CALIBNET.FUSION_DECODER.NAME
    return MASK_BRANCH_REGISTRY.get(name)(cfg)