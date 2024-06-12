import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

from .position_encoding import build_position_encoding
from .ops.modules import MSDeformAttn
from .trans_utils import _get_activation_fn, _get_clones

CALIBNET_ENCODER_REGISTRY = Registry("CALIBNET_ENCODER")
CALIBNET_ENCODER_REGISTRY.__doc__ = "registry for CalibNet encoder"


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src

class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, n_head=8, num_layers=6, dim_feedforward=1024,
                    dropout=0.1, activation="relu", num_feature_levels=4, n_points=4,):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, n_head, n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
        
    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # [bs, 2835, d]
        mask_flatten = torch.cat(mask_flatten, 1)  # [bs, 2835]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # [bs, 2835, d]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


@CALIBNET_ENCODER_REGISTRY.register()
class TransformerEncoder(nn.Module):
    def __init__(self, cfg, input_shape, is_depth=False):
        super().__init__()
        if is_depth:
            self.num_channels = cfg.MODEL.CALIBNET.ENCODER_DEPTH.NUM_CHANNELS
            self.in_features = cfg.MODEL.CALIBNET.ENCODER_DEPTH.IN_FEATURES
        else:
            self.num_channels = cfg.MODEL.CALIBNET.ENCODER.NUM_CHANNELS
            self.in_features = cfg.MODEL.CALIBNET.ENCODER.IN_FEATURES
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        self.num_trans_lvls = len(self.in_channels)
        self.input_convs = nn.ModuleList()
        for in_channel in self.in_channels[::-1]:
            input_conv = nn.Sequential(
                nn.Conv2d(in_channel, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, self.num_channels),
                nn.ReLU(inplace=True),
            )
            c2_xavier_fill(input_conv[0])
            self.input_convs.append(input_conv)
        self.res2_conv = nn.Sequential(
                nn.Conv2d(input_shape['res2'].channels, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, self.num_channels),
                nn.ReLU(inplace=True),
            )
        c2_xavier_fill(self.res2_conv[0])

        if is_depth:
            self.trans_encoder = MSDeformAttnTransformerEncoderOnly(
                d_model=self.num_channels,
                dropout=cfg.MODEL.CALIBNET.ENCODER_DEPTH.DROPOUT,
                n_head=cfg.MODEL.CALIBNET.ENCODER_DEPTH.NUM_HEADS,
                dim_feedforward=cfg.MODEL.CALIBNET.ENCODER_DEPTH.DIM_FEEDFORWARD,
                num_layers=cfg.MODEL.CALIBNET.ENCODER_DEPTH.NUM_LAYERS,
                activation=cfg.MODEL.CALIBNET.ENCODER_DEPTH.ACTIVATION,
                num_feature_levels=self.num_trans_lvls
            )
        else:
            self.trans_encoder = MSDeformAttnTransformerEncoderOnly(
                d_model=self.num_channels,
                dropout=cfg.MODEL.CALIBNET.ENCODER.DROPOUT,
                n_head=cfg.MODEL.CALIBNET.ENCODER.NUM_HEADS,
                dim_feedforward=cfg.MODEL.CALIBNET.ENCODER.DIM_FEEDFORWARD,
                num_layers=cfg.MODEL.CALIBNET.ENCODER.NUM_LAYERS,
                activation=cfg.MODEL.CALIBNET.ENCODER.ACTIVATION,
                num_feature_levels=self.num_trans_lvls
            )
        self.pe_layer = build_position_encoding(self.num_channels)

    def forward(self, features):
        srcs = []
        pos = []
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f].float()
            srcs.append(self.input_convs[idx](x))
            pos.append(self.pe_layer(x))
        # transformer encoder
        memory, spatial_shapes, level_start_index = self.trans_encoder(srcs, pos)  # memory:[bs, hw_total, d]
        # split features
        bs = memory.shape[0]
        split_size_or_sections = [None] * self.num_trans_lvls
        for i in range(self.num_trans_lvls):
            if i < self.num_trans_lvls - 1:
                split_size_or_sections[i] = level_start_index[i+1]-level_start_index[i]
            else:
                split_size_or_sections[i] = memory.shape[1] - level_start_index[i]
        memory = torch.split(memory, split_size_or_sections, dim=1)
        out = []
        for i , feat in enumerate(memory):
            out.append(feat.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        out.append(self.res2_conv(features['res2'].float()))
        return out



def build_calibnet_encoder(cfg, input_shape):
    name = cfg.MODEL.CALIBNET.ENCODER.NAME
    return CALIBNET_ENCODER_REGISTRY.get(name)(cfg, input_shape, is_depth=False)

def build_calibnet_encoder_depth(cfg, input_shape):
    name = cfg.MODEL.CALIBNET.ENCODER_DEPTH.NAME
    return CALIBNET_ENCODER_REGISTRY.get(name)(cfg, input_shape, is_depth=True)