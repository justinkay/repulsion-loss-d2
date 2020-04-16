import numpy as np
import torch

from detectron2.layers import cat
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures import Boxes, pairwise_iou


def iog(gt, pred):
    """
    Intersection over ground truth.
    """
    inter_xmin = torch.max(gt[:, 0], pred[:, 0])
    inter_ymin = torch.max(gt[:, 1], pred[:, 1])
    inter_xmax = torch.min(gt[:, 2], pred[:, 2])
    inter_ymax = torch.min(gt[:, 3], pred[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    
    # handle empty boxes
    iog = torch.where(
        (I > 0) & (G > 0),
        I / G,
        torch.zeros(1, dtype=gt.dtype, device=gt.device),
    )

    return iog

def smooth_ln(x, sigma):
    return torch.where(
        torch.le(x, sigma),
        -torch.log(1 - x),
        ((x - sigma) / (1 - sigma)) - np.log(1 - sigma)
    )

class RepLossFastRCNNOutputs(FastRCNNOutputs):

    def __init__(self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta):
        super().__init__(box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta)
        if proposals[0].has("gt_boxes"):
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            self.gt_box_inds = cat([p.gt_box_inds for p in proposals], dim=0)

    def rep_gt_loss(self):
        # get all predicted boxes in this minibatch; meaning all proposals + all predicted box regression deltas
        box_preds = self.predict_boxes_all()
        repulsion_targets = self.gt_rep_boxes.tensor

        # get boxes of interest: foreground boxes with a repulsion target
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_mask = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        valid_mask = self.gt_rep_boxes.area() > 0
        inds = torch.nonzero((fg_mask) & (valid_mask)).squeeze(1)
        box_preds = box_preds[inds]
        repulsion_targets = repulsion_targets[inds]

        # calculate loss
        # as in FastRCNNOutputs:smooth_l1_loss, divide by total examples instead of total
        # foreground examples to weight each foreground example the same
        iogs = self.IoG(repulsion_targets, box_preds)
        losses = self.smooth_ln(iogs, sigma=0.9)
        loss_rep_gt = torch.sum(losses) / self.gt_classes.numel()

        print("loss_rep_gt", loss_rep_gt)
        return loss_rep_gt

    def rep_box_loss(self):
        # get positive (foreground) proposals (P+ in RepLoss paper)
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )

        # IOUs here also deal with regressed boxes
        boxes = Boxes(self.predict_boxes_all())

        # set of regressed boxes of positive proposals
        fg_boxes = boxes[fg_inds]

        # index of ground truth box for each positive proposal
        fg_gt_inds = self.gt_box_inds[fg_inds]

        # rep box loss is sum of IOUs of boxes with different GT
        # targets
        num_gts = torch.max(self.gt_box_inds) + 1
        sum = torch.tensor(0.0)
        for i in range(num_gts):
            boxes_i = fg_boxes[fg_gt_inds == i]
            for j in range(num_gts):
                boxes_j = fg_boxes[fg_gt_inds == j]
                if i != j:
                    iou_matrix = pairwise_iou(boxes_i, boxes_j)
                    losses = smooth_ln(iou_matrix, sigma=0.1)
                    sum += torch.sum(iou_matrix)
        
        # every i,j was counted twice
        sum /= 2.0

        # as in FastRCNNOutputs:smooth_l1_loss, divide by total examples instead of total
        # foreground examples to weight each foreground example the same
        loss_rep_box = sum / self.gt_classes.numel()

        print("loss_rep_box", loss_rep_box)
        return loss_rep_box

    def repulsion_loss(self):
        # TODO
        a=0.5
        b=0.5
        
        # note that all loss components normalize using the total number of regions, rather than the
        # number of foreground (positive proposal) regions - see explanation in smooth_l1_loss()
        loss_attr = self.smooth_l1_loss()
        loss_rep_gt = self.rep_gt_loss()
        loss_rep_box = self.rep_box_loss()

        repulsion_loss = loss_attr +  a*loss_rep_gt + b*loss_rep_box
        return repulsion_loss

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.repulsion_loss(),
        }

    def predict_boxes_all(self):
        """
        Predictions of all proposal regions in this minibatch.
        So, proposals + proposal_deltas.
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes