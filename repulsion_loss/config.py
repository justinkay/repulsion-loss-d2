from detectron2.config import CfgNode as CN


def add_reploss_config(cfg):
    """
    Add config for ROI head with Repulsion Loss.
    """
    _C = cfg

    _C.MODEL.ROI_HEADS.REPULSION_LOSS = CN()
    _C.MODEL.ROI_HEADS.REPULSION_LOSS.REP_GT_FACTOR = 0.5 # 'alpha' in paper
    _C.MODEL.ROI_HEADS.REPULSION_LOSS.REP_BOX_FACTOR = 0.5 # 'beta' in paper
    _C.MODEL.ROI_HEADS.REPULSION_LOSS.REP_GT_SIGMA = 0.9 # for smooth_ln
    _C.MODEL.ROI_HEADS.REPULSION_LOSS.REP_BOX_SIGMA = 0.1 # for smooth_ln
    _C.MODEL.ROI_HEADS.REPULSION_LOSS.D2_NORMALIZE = True # False to normalize like the paper; see RepLossFastRCNNOutputs