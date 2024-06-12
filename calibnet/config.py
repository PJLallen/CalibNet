from detectron2.config import CfgNode as CN

def add_calibnet_config(cfg):
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.MASK_ON = True
    # CalibNet
    cfg.MODEL.CALIBNET = CN()
    cfg.MODEL.CALIBNET.NUM_CLASSES = 1
    # parameters for inference
    cfg.MODEL.CALIBNET.CLS_THRESHOLD = 0.005
    cfg.MODEL.CALIBNET.MASK_THRESHOLD = 0.45
    cfg.MODEL.CALIBNET.MAX_DETECTIONS = 50
    # dataset mapper
    cfg.MODEL.CALIBNET.DATASET_MAPPER = "RGBDSISDatasetMapper"

    # [model]
    # encoder
    cfg.MODEL.CALIBNET.ENCODER = CN()
    cfg.MODEL.CALIBNET.ENCODER.NAME = "TransformerEncoder"
    cfg.MODEL.CALIBNET.ENCODER.NUM_CHANNELS = 256
    cfg.MODEL.CALIBNET.ENCODER.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.CALIBNET.ENCODER.DROPOUT = 0.0
    cfg.MODEL.CALIBNET.ENCODER.NUM_HEADS = 8
    cfg.MODEL.CALIBNET.ENCODER.DIM_FEEDFORWARD = 1024
    cfg.MODEL.CALIBNET.ENCODER.NUM_LAYERS = 3
    cfg.MODEL.CALIBNET.ENCODER.ACTIVATION = "gelu"
    # encoder depth
    cfg.MODEL.CALIBNET.ENCODER_DEPTH = CN()
    cfg.MODEL.CALIBNET.ENCODER_DEPTH.NAME = "TransformerEncoder"
    cfg.MODEL.CALIBNET.ENCODER_DEPTH.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.CALIBNET.ENCODER_DEPTH.NUM_CHANNELS = 256
    cfg.MODEL.CALIBNET.ENCODER_DEPTH.DROPOUT = 0.0
    cfg.MODEL.CALIBNET.ENCODER_DEPTH.NUM_HEADS = 8
    cfg.MODEL.CALIBNET.ENCODER_DEPTH.DIM_FEEDFORWARD = 1024
    cfg.MODEL.CALIBNET.ENCODER_DEPTH.NUM_LAYERS = 3
    cfg.MODEL.CALIBNET.ENCODER_DEPTH.ACTIVATION = "gelu"
    # decoder
    cfg.MODEL.CALIBNET.DECODER = CN()
    cfg.MODEL.CALIBNET.DECODER.NAME = "MaskfeautreIAMDecoder"
    cfg.MODEL.CALIBNET.DECODER.NUM_KERNELS = 100
    cfg.MODEL.CALIBNET.DECODER.INPUT_CHANNELS = 256
    cfg.MODEL.CALIBNET.DECODER.CONV_DIM = 256
    cfg.MODEL.CALIBNET.DECODER.SCALE_FACTOR = 2
    # fusion decoder
    cfg.MODEL.CALIBNET.FUSION_DECODER = CN()
    cfg.MODEL.CALIBNET.FUSION_DECODER.NAME = "VanillaFusionDecoder"
    cfg.MODEL.CALIBNET.FUSION_DECODER.INPUT_DIM = 256
    cfg.MODEL.CALIBNET.FUSION_DECODER.MASK_DIM = 256
    cfg.MODEL.CALIBNET.FUSION_DECODER.INPUT_STRIDES = [32,16,8]
    cfg.MODEL.CALIBNET.FUSION_DECODER.NORM = "GN"
    # pvt backbone
    cfg.MODEL.PVTV2 = CN()
    cfg.MODEL.PVTV2.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.PVTV2.NAME = "b2"
    cfg.MODEL.PVTV2.LINEAR = True
    # p2t backbone
    cfg.MODEL.P2T = CN()
    cfg.MODEL.P2T.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.P2T.NAME = "large"
    # swin
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    # [loss]
    cfg.MODEL.CALIBNET.LOSS = CN()
    cfg.MODEL.CALIBNET.LOSS.NAME = "CalibNetCriterion"
    cfg.MODEL.CALIBNET.LOSS.ITEMS = ("labels", "masks")
    # loss weight
    cfg.MODEL.CALIBNET.LOSS.CLASS_WEIGHT = 2.0
    cfg.MODEL.CALIBNET.LOSS.MASK_PIXEL_WEIGHT = 5.0
    cfg.MODEL.CALIBNET.LOSS.MASK_DICE_WEIGHT = 2.0
    # iou-aware objectness loss weight
    cfg.MODEL.CALIBNET.LOSS.OBJECTNESS_WEIGHT = 1.0
    # Matcher
    cfg.MODEL.CALIBNET.MATCHER = CN()
    cfg.MODEL.CALIBNET.MATCHER.NAME = "CalibNetMatcher"
    cfg.MODEL.CALIBNET.MATCHER.ALPHA = 0.8
    cfg.MODEL.CALIBNET.MATCHER.BETA = 0.2

    # [Optimizer]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.AMSGRAD = False

    # [Pyramid Vision Transformer]
    cfg.MODEL.PVT = CN()
    cfg.MODEL.PVT.NAME = "b1"
    cfg.MODEL.PVT.OUT_FEATURES = ["p2", "p3", "p4"]
    cfg.MODEL.PVT.LINEAR = False

    cfg.MODEL.CSPNET = CN()
    cfg.MODEL.CSPNET.NAME = "darknet53"
    cfg.MODEL.CSPNET.NORM = ""
    # (csp-)darknet: csp1, csp2, csp3, csp4
    cfg.MODEL.CSPNET.OUT_FEATURES = ["csp1", "csp2", "csp3", "csp4"]

