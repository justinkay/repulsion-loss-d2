import numpy as np
import torch

from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from .fast_rcnn import RepLossFastRCNNOutputs
from .matcher import Top2Matcher


@ROI_HEADS_REGISTRY.register()
class RepLossROIHeads(StandardROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        
        self.proposal_matcher = Top2Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`StandardROIHeads.forward`.

        Only included here because there are new calls to
            self.label_and_sample_proposals, and
            self._forward_boxes
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            losses.update(self._forward_mask(features_list, proposals))
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        See :class:`StandardROIHeads._forward_box`.

        Only change is that RepLossFastRCNNOutputs is called.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes", "gp_rep_boxes", "gt_box_inds"

        Returns:
            In training, a dict of losses.
            In inference, a listf of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = RepLossFastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                - gt_box_inds - index in targets of each box in gt_boxes
                - gt_rep_boxes - repulsion target for each proposal

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            has_repel_target = len(targets_per_image) > 1

            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels, repel_matched_idxs = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            sampled_targets = matched_idxs[sampled_idxs]

            # also store the index of the target box - useful for creating subsets later
            proposals_per_image.set("gt_box_inds", sampled_targets)

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            # add in the repulsion loss info - store the box we are repelling away from
            # if no repulsion target, use a box { 0, 0, 0, 0 } like for attraction target
            if has_repel_target:
                sampled_repels = repel_matched_idxs[sampled_idxs]
                proposals_per_image.set("gt_rep_boxes", targets_per_image.gt_boxes[sampled_repels])
            else:
                gt_rep_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.set("gt_rep_boxes", gt_rep_boxes)

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt