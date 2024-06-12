import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from calibnet.loss import build_calibnet_criterion
from calibnet.utils import nested_tensor_from_tensor_list

from calibnet.trans_encoder.encoder import build_calibnet_encoder, build_calibnet_encoder_depth
from calibnet.kernel_branch.calibnet_decoder import build_kernel_branch
from calibnet.mask_branch.calibnet_fusion_decoder import build_mask_branch
from calibnet.kernel_branch.decoder_utils import CoarsePredictor

__all__ = ["CalibNet"]


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


@META_ARCH_REGISTRY.register()
class CalibNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # move to target device
        self.device = torch.device(cfg.MODEL.DEVICE)

        # backbone
        self.backbone = build_backbone(cfg)
        self.backbonedepth = copy.deepcopy(self.backbone)
        self.size_divisibility = self.backbone.size_divisibility
        output_shape = self.backbone.output_shape()

        # encoder & decoder
        self.encoder = build_calibnet_encoder(cfg, output_shape)
        self.encoder_depth = build_calibnet_encoder_depth(cfg, output_shape)

        self.decoder = build_kernel_branch(cfg)
        self.fusion_decoder = build_mask_branch(cfg)

        # matcher & loss (matcher is built in loss)
        self.criterion = build_calibnet_criterion(cfg)

        # data and preprocessing
        self.mask_format = cfg.INPUT.MASK_FORMAT

        self.pixel_mean = torch.Tensor(
            cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(
            cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # inference
        self.cls_threshold = cfg.MODEL.CALIBNET.CLS_THRESHOLD
        self.mask_threshold = cfg.MODEL.CALIBNET.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.CALIBNET.MAX_DETECTIONS

        feat_dim = cfg.MODEL.CALIBNET.DECODER.INPUT_CHANNELS
        self.predictor_rgb = CoarsePredictor(feat_dim)
        self.predictor_depth = CoarsePredictor(feat_dim)
        self.predictor_rgb_r2 = CoarsePredictor(feat_dim)
        self.predictor_depth_d2 = CoarsePredictor(feat_dim)

    def normalizer(self, image):
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    def preprocess_inputs(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, 32)
        depths = [x["depth"].to(self.device) for x in batched_inputs]
        depths = [self.normalizer(x) for x in depths]
        depths = ImageList.from_tensors(depths, 32)
        return images, depths

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            gt_classes = targets_per_image.gt_classes
            target["labels"] = gt_classes.to(self.device)
            h, w = targets_per_image.image_size
            if not targets_per_image.has('gt_masks'):
                gt_masks = BitMasks(torch.empty(0, h, w))
            else:
                gt_masks = targets_per_image.gt_masks
                if self.mask_format == "polygon":
                    if len(gt_masks.polygons) == 0:
                        gt_masks = BitMasks(torch.empty(0, h, w))
                    else:
                        gt_masks = BitMasks.from_polygon_masks(
                            gt_masks.polygons, h, w)
            binary_gt_mask = (torch.sum(gt_masks.tensor,dim=0)>0).unsqueeze(0)
            target["masks"] = gt_masks.to(self.device)
            target["binary_mask"] = binary_gt_mask.to(self.device)
            new_targets.append(target)

        return new_targets

    def forward(self, batched_inputs):
        images, depths = self.preprocess_inputs(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        if isinstance(depths, (list, torch.Tensor)):
            depths = nested_tensor_from_tensor_list(depths)
        max_shape = images.tensor.shape[2:]
        # forward
        features = self.backbone(images.tensor)
        features_depth = self.backbonedepth(depths.tensor)

        features = self.encoder(features) 
        features_depth = self.encoder_depth(features_depth)

        mask_feature = self.fusion_decoder(features, features_depth)

        output = self.decoder(features[2], features_depth[2], mask_feature)  


        if self.training:
            # coarse predictor
            rgb_pred = self.predictor_rgb(features[2])
            depth_pred = self.predictor_depth(features_depth[2])
            rgb_pred_r2 = self.predictor_rgb_r2(features[3])
            depth_pred_d2 = self.predictor_depth_d2(features_depth[3])
            # add aux_result
            output['rgb_pred'] = rgb_pred
            output['depth_pred'] = depth_pred
            output['rgb_pred_r2'] = rgb_pred_r2
            output['depth_pred_d2'] = depth_pred_d2
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            losses = self.criterion(output, targets, max_shape)
            return losses
        else:
            results = self.inference(
                output, batched_inputs, max_shape, images.image_sizes)
            processed_results = [{"instances": r} for r in results]
            return processed_results

    def forward_test(self, images):
        images = (images - self.pixel_mean[None]) / self.pixel_std[None]
        features = self.backbone(images)
        features = self.encoder(features)
        output = self.decoder(features)

        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_objectness = output["pred_scores"].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)
        pred_masks = F.interpolate(
            pred_masks, scale_factor=4.0, mode="bilinear", align_corners=False)
        return pred_scores, pred_masks

    def inference(self, output, batched_inputs, max_shape, image_sizes):
        results = []
        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_objectness = output["pred_scores"].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)

        for _, (scores_per_image, mask_pred_per_image, batched_input, img_shape) in enumerate(zip(
                pred_scores, pred_masks, batched_inputs, image_sizes)):

            ori_shape = (batched_input["height"], batched_input["width"])
            result = Instances(ori_shape)
            scores, labels = scores_per_image.max(dim=-1)
            keep = scores > self.cls_threshold
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = mask_pred_per_image[keep]

            if scores.size(0) == 0:
                result.scores = scores
                result.pred_classes = labels
                results.append(result)
                continue

            h, w = img_shape
            scores = rescoring_mask(
                scores, mask_pred_per_image > self.mask_threshold, mask_pred_per_image)

            mask_pred_per_image = F.interpolate(
                mask_pred_per_image.unsqueeze(1), size=max_shape, mode="bilinear", align_corners=False)[:, :, :h, :w]
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image, size=ori_shape, mode='bilinear', align_corners=False).squeeze(1)

            mask_pred = mask_pred_per_image > self.mask_threshold
            result.pred_masks = mask_pred
            result.scores = scores
            result.pred_classes = labels
            results.append(result)

        return results






