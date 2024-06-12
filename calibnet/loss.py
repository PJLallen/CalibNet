import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.utils.registry import Registry

from .utils import nested_masks_from_list, is_dist_avail_and_initialized, get_world_size

CALIBNET_MATCHER_REGISTRY = Registry("CalibNet_MATCHER")
CALIBNET_MATCHER_REGISTRY.__doc__ = "Matcher for CalibNet"
CALIBNET_CRITERION_REGISTRY = Registry("CalibNet_CRITERION")
CALIBNET_CRITERION_REGISTRY.__doc__ = "Criterion for CalibNet"


def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()


@CALIBNET_CRITERION_REGISTRY.register()
class CalibNetCriterion(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py

    def __init__(self, cfg, matcher):
        super().__init__()
        self.matcher = matcher
        self.losses = cfg.MODEL.CALIBNET.LOSS.ITEMS
        self.weight_dict = self.get_weight_dict(cfg)
        self.num_classes = cfg.MODEL.CALIBNET.NUM_CLASSES

    def get_weight_dict(self, cfg):
        losses = ("loss_ce", "loss_mask", "loss_dice", "loss_objectness", "loss_t3", "loss_d3", "loss_r2", "loss_d2")
        weight_dict = {}
        ce_weight = cfg.MODEL.CALIBNET.LOSS.CLASS_WEIGHT
        mask_weight = cfg.MODEL.CALIBNET.LOSS.MASK_PIXEL_WEIGHT
        dice_weight = cfg.MODEL.CALIBNET.LOSS.MASK_DICE_WEIGHT
        objectness_weight = cfg.MODEL.CALIBNET.LOSS.OBJECTNESS_WEIGHT
        t3_weight = 0.5
        d3_weight = 0.5

        weight_dict = dict(
            zip(losses, (ce_weight, mask_weight, dice_weight, objectness_weight, t3_weight, d3_weight, 0.5, 0.2)))
        return weight_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_instances, input_shape=None):
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                     for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(
            target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        class_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ) / num_instances
        losses = {'loss_ce': class_loss}
        return losses

    def loss_masks_with_iou_objectness(self, outputs, targets, indices, num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # Bx100xHxW
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]
        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
        num_masks = [len(t["masks"]) for t in targets]
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:
            losses = {
                "loss_dice": src_masks.sum() * 0.0,
                "loss_mask": src_masks.sum() * 0.0,
                "loss_objectness": src_iou_scores.sum() * 0.0
            }
            return losses

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(
            target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)
        # FIXME: tgt_idx
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        with torch.no_grad():
            ious = compute_mask_iou(src_masks, target_masks)

        tgt_iou_scores = ious
        src_iou_scores = src_iou_scores[src_idx]
        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
            "loss_mask": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        }
        return losses

    def loss_binary(self, outputs, targets, input_shape):
        # prepare predictions
        pred_t3 = outputs['rgb_pred'][0]
        pred_d3 = outputs['depth_pred'][0]
        pred_r2 = outputs['rgb_pred_r2'][0]
        pred_d2 = outputs['depth_pred_d2'][0]
        pred_t3 = F.interpolate(pred_t3, scale_factor=2, mode='bilinear', align_corners=False)
        pred_d3 = F.interpolate(pred_d3, scale_factor=2, mode='bilinear', align_corners=False)
        # prepare gt
        with torch.no_grad():
            target_mask, _ = nested_masks_from_list(
                [t["binary_mask"] for t in targets], input_shape).decompose()
        target_mask = target_mask.float().unsqueeze(1)
        target_mask = F.interpolate(target_mask, scale_factor=0.25, mode='bilinear', align_corners=False)
        losses = {
            "loss_t3": F.binary_cross_entropy_with_logits(pred_t3, target_mask, reduction='mean'),
            "loss_d3": F.binary_cross_entropy_with_logits(pred_d3, target_mask, reduction='mean'),
            "loss_r2": F.binary_cross_entropy_with_logits(pred_r2, target_mask, reduction='mean'),
            "loss_d2": F.binary_cross_entropy_with_logits(pred_d2, target_mask, reduction='mean'),
        }
        return losses


    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks_with_iou_objectness,
        }
        if loss == "loss_objectness":
            return {}
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    def forward(self, outputs, targets, input_shape):

        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, input_shape)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(
            num_instances / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_instances, input_shape=input_shape))

        losses.update(self.loss_binary(outputs, targets, input_shape=input_shape))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses


@CALIBNET_MATCHER_REGISTRY.register()
class CalibNetMatcher(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.MODEL.CALIBNET.MATCHER.ALPHA
        self.beta = cfg.MODEL.CALIBNET.MATCHER.BETA
        self.mask_score = dice_score

    def forward(self, outputs, targets, input_shape):
        with torch.no_grad():
            B, N, H, W = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits'].sigmoid()

            tgt_ids = torch.cat([v["labels"] for v in targets])

            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
            tgt_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
            device = pred_masks.device
            tgt_masks = tgt_masks.to(pred_masks)

            tgt_masks = F.interpolate(
                tgt_masks[:, None], size=pred_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            pred_masks = pred_masks.view(B * N, -1)
            tgt_masks = tgt_masks.flatten(1)
            with autocast(enabled=False):
                pred_masks = pred_masks.float()
                tgt_masks = tgt_masks.float()
                pred_logits = pred_logits.float()
                mask_score = self.mask_score(pred_masks, tgt_masks)
                # Nx(Number of gts)
                matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]
                C = (mask_score ** self.alpha) * (matching_prob ** self.beta)

            C = C.view(B, N, -1).cpu()
            # hungarian matching
            sizes = [len(v["masks"]) for v in targets]
            indices = [linear_sum_assignment(c[i], maximize=True)
                       for i, c in enumerate(C.split(sizes, -1))]
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
                j, dtype=torch.int64)) for i, j in indices]
            return indices


def build_calibnet_matcher(cfg):
    name = cfg.MODEL.CALIBNET.MATCHER.NAME
    return CALIBNET_MATCHER_REGISTRY.get(name)(cfg)


def build_calibnet_criterion(cfg):
    matcher = build_calibnet_matcher(cfg)
    name = cfg.MODEL.CALIBNET.LOSS.NAME
    return CALIBNET_CRITERION_REGISTRY.get(name)(cfg, matcher)
