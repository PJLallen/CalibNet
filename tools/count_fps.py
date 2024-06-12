import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, print_csv_format

sys.path.append(".")

from calibnet import COCOMaskEvaluator
from calibnet import add_calibnet_config, RGBDSISTestDatasetMapper
from calibnet.trans_encoder.encoder import build_calibnet_encoder, build_calibnet_encoder_depth
from calibnet.kernel_branch.calibnet_decoder import build_kernel_branch
from calibnet.mask_branch.calibnet_fusion_decoder import build_mask_branch

device = torch.device('cuda:0')
dtype = torch.float32

__all__ = ["CalibNet"]

pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).to(device).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).to(device).view(3, 1, 1)


@torch.jit.script
def normalizer(x, mean, std): return (x - mean) / std


def synchronize():
    torch.cuda.synchronize()


def process_batched_inputs(batched_inputs):
    images = [x["image"].to(device) for x in batched_inputs]
    images = [normalizer(x, pixel_mean, pixel_std) for x in images]
    images = ImageList.from_tensors(images, 32)
    depths = [x["depth"].to(device) for x in batched_inputs]
    depths = [normalizer(x, pixel_mean, pixel_std) for x in depths]
    depths = ImageList.from_tensors(depths, 32)
    ori_size = (batched_inputs[0]["height"], batched_inputs[0]["width"])
    return images.tensor, depths.tensor, images.image_sizes[0], ori_size


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))

class CalibNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.backbonedepth = copy.deepcopy(self.backbone)
        self.size_divisibility = self.backbone.size_divisibility
        output_shape = self.backbone.output_shape()
        # encoder & decoder
        self.encoder = build_calibnet_encoder(cfg, output_shape)
        self.encoder_depth = build_calibnet_encoder_depth(cfg, output_shape)
        self.decoder = build_kernel_branch(cfg)
        self.fusion_decoder = build_mask_branch(cfg)
        self.to(self.device)
        # inference
        self.cls_threshold = cfg.MODEL.CALIBNET.CLS_THRESHOLD
        self.mask_threshold = cfg.MODEL.CALIBNET.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.CALIBNET.MAX_DETECTIONS
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.num_classes = cfg.MODEL.CALIBNET.NUM_CLASSES

    def forward(self, image, depth, resized_size, ori_size):
        max_size = image.shape[2:]
        features = self.backbone(image)
        features_depth = self.backbonedepth(depth)

        features = self.encoder(features)
        features_depth = self.encoder_depth(features_depth)
        mask_feature = self.fusion_decoder(features, features_depth)
        output = self.decoder(features[2], features_depth[2], mask_feature) 
        
        result = self.inference_single(
            output, resized_size, max_size, ori_size)
        return result

    def inference_single(self, outputs, img_shape, pad_shape, ori_shape):
        """
        inference for only one sample
        Args:
            scores (tensor): [NxC]
            masks (tensor): [NxHxW]
            img_shape (list): (h1, w1), image after resized
            pad_shape (list): (h2, w2), padded resized image
            ori_shape (list): (h3, w3), original shape h3*w3 < h1*w1 < h2*w2
        """
        result = Instances(ori_shape)
        # scoring
        pred_logits = outputs["pred_logits"][0].sigmoid()
        pred_scores = outputs["pred_scores"][0].sigmoid().squeeze()
        pred_masks = outputs["pred_masks"][0].sigmoid()
        # obtain scores
        scores, labels = pred_logits.max(dim=-1)
        # remove by thresholding
        keep = scores > self.cls_threshold
        scores = torch.sqrt(scores[keep] * pred_scores[keep])
        labels = labels[keep]
        pred_masks = pred_masks[keep]

        if scores.size(0) == 0:
            return None
        scores = rescoring_mask(scores, pred_masks > 0.45, pred_masks)
        h, w = img_shape
        # resize masks
        pred_masks = F.interpolate(pred_masks.unsqueeze(1), size=pad_shape,
                                   mode="bilinear", align_corners=False)[:, :, :h, :w]
        pred_masks = F.interpolate(pred_masks, size=ori_shape, mode='bilinear',
                                   align_corners=False).squeeze(1)
        mask_pred = pred_masks > self.mask_threshold

        mask_pred = BitMasks(mask_pred)
        result.pred_masks = mask_pred
        result.scores = scores
        result.pred_classes = labels
        return result
    

def test_calibnet_speed(cfg, fp16=False):
    device = torch.device('cuda:0')


    model = CalibNet(cfg)
    model.eval()
    model.to(device)
    print(model)
    size = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False)

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = False

    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    evaluator = COCOMaskEvaluator(
        cfg.DATASETS.TEST[0], ("segm",), False, output_folder)
    evaluator.reset()
    model.to(device)
    model.eval()
    mapper = RGBDSISTestDatasetMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    durations = []

    with autocast(enabled=fp16):
        with torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                images, depths, resized_size, ori_size = process_batched_inputs(inputs)
                synchronize()
                start_time = time.perf_counter()
                output = model(images, depths, resized_size, ori_size)
                synchronize()
                end = time.perf_counter() - start_time

                durations.append(end)
                if idx % 200 == 0:
                    print("process: [{}/{}] fps: {:.3f}".format(idx,
                                                                len(data_loader), 1/np.mean(durations[100:])))
                evaluator.process(inputs, [{"instances": output}])
    # evaluate
    # for(int i = 0)
    results = evaluator.evaluate()
    print_csv_format(results)

    latency = np.mean(durations[100:])
    fps = 1 / latency
    print("speed: {:.4f}s FPS: {:.2f}".format(latency, fps))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_calibnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == '__main__':

    args = default_argument_parser()
    args.add_argument("--fp16", action="store_true",
                      help="support fp16 for inference")
    args = args.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    test_calibnet_speed(cfg, fp16=args.fp16)
