# Copyright (c) OpenMMLab. All rights reserved.
import lap
import numpy as np
import torch
from addict import Dict
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.core.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from mmtrack.models import TRACKERS
from .sort_tracker import SortTracker


@TRACKERS.register_module()
class OCSORTTracker(SortTracker):
    """Tracker for OC-SORT.

    Args:
        obj_score_thrs (float): Detection score threshold for matching objects.
            Defaults to 0.3.
        init_track_thr (float): Detection score threshold for initializing a
            new tracklet. Defaults to 0.7.
        weight_iou_with_det_scores (bool): Whether using detection scores to
            weight IOU which is used for matching. Defaults to True.
        match_iou_thr (float): IOU distance threshold for matching between two
            frames. Defaults to 0.3.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
        vel_consist_weight (float): Weight of the velocity consistency term in
            association (OCM term in the paper).
        vel_delta_t (int): The difference of time step for calculating of the
            velocity direction of tracklets.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 obj_score_thr=0.3,
                 init_track_thr=0.7,
                 weight_iou_with_det_scores=True,
                 match_iou_thr=0.3,
                 num_tentatives=3,
                 vel_consist_weight=0.2,
                 vel_delta_t=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thr = obj_score_thr
        self.init_track_thr = init_track_thr

        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.match_iou_thr = match_iou_thr
        self.vel_consist_weight = vel_consist_weight
        self.vel_delta_t = vel_delta_t

        self.num_tentatives = num_tentatives

    @property
    def unconfirmed_ids(self):
        """Unconfirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if track.tentative]
        return ids

    def init_track(self, id, obj):
        """Initialize a track."""
        super().init_track(id, obj)
        if self.tracks[id].frame_ids[-1] == 0:
            self.tracks[id].tentative = False
        else:
            self.tracks[id].tentative = True
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)
        # track.obs maintains the history associated detections to this track
        self.tracks[id].obs = []
        bbox_id = self.memo_items.index('bboxes')
        self.tracks[id].obs.append(obj[bbox_id])
        # a placefolder to save mean/covariance before losing tracking it
        # parameters to save: mean, covariance, measurement
        self.tracks[id].tracked = True
        self.tracks[id].saved_attr = Dict()
        self.tracks[id].velocity = torch.tensor(
            (-1, -1)).to(obj[bbox_id].device)  # placeholder

    def update_track(self, id, obj):
        """Update a track."""
        super().update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)
        self.tracks[id].tracked = True
        bbox_id = self.memo_items.index('bboxes')
        self.tracks[id].obs.append(obj[bbox_id])

        bbox1 = self.k_step_observation(self.tracks[id])
        bbox2 = obj[bbox_id]
        self.tracks[id].velocity = self.vel_direction(bbox1, bbox2).to(
            obj[bbox_id].device)

    def vel_direction(self, bbox1, bbox2):
        """Estimate the direction vector between two boxes."""
        if bbox1.sum() < 0 or bbox2.sum() < 0:
            return torch.tensor((-1, -1))
        cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
        cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
        speed = torch.tensor([cy2 - cy1, cx2 - cx1])
        norm = torch.sqrt((speed[0])**2 + (speed[1])**2) + 1e-6
        return speed / norm

    def vel_direction_batch(self, bboxes1, bboxes2):
        """Estimate the direction vector given two batches of boxes."""
        cx1, cy1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2.0, (bboxes1[:, 1] +
                                                           bboxes1[:, 3]) / 2.0
        cx2, cy2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2.0, (bboxes2[:, 1] +
                                                           bboxes2[:, 3]) / 2.0
        speed_diff_y = cy2[None, :] - cy1[:, None]
        speed_diff_x = cx2[None, :] - cx1[:, None]
        speed = torch.cat((speed_diff_y[..., None], speed_diff_x[..., None]),
                          dim=-1)
        norm = torch.sqrt((speed[:, :, 0])**2 + (speed[:, :, 1])**2) + 1e-6
        speed[:, :, 0] /= norm
        speed[:, :, 1] /= norm
        return speed

    def k_step_observation(self, track):
        """return the observation k step away before."""
        obs_seqs = track.obs
        num_obs = len(obs_seqs)
        if num_obs == 0:
            return torch.tensor((-1, -1, -1, -1)).to(track.obs[0].device)
        elif num_obs > self.vel_delta_t:
            if obs_seqs[num_obs - 1 - self.vel_delta_t] is not None:
                return obs_seqs[num_obs - 1 - self.vel_delta_t]
            else:
                return self.last_obs(track)
        else:
            return self.last_obs(track)

    def ocm_assign_ids(self,
                       ids,
                       det_bboxes,
                       weight_iou_with_det_scores=False,
                       match_iou_thr=0.5):
        """Apply Observation-Centric Momentum (OCM) to assign ids.

        OCM adds movement direction consistency into the association cost
        matrix. This term requires no additional assumption but from the
        same linear motion assumption as the canonical Kalman Filter in SORT.

        Args:
            ids (list[int]): Tracking ids.
            det_bboxes (Tensor): of shape (N, 5)
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.

        Returns:
            tuple(int): The assigning ids.

        OC-SORT uses velocity consistency besides IoU for association
        """
        # get track_bboxes
        track_bboxes = np.zeros((0, 4))
        for id in ids:
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

        # compute distance
        ious = bbox_overlaps(track_bboxes, det_bboxes[:, :4])
        if weight_iou_with_det_scores:
            ious *= det_bboxes[:, 4][None]
        dists = (1 - ious).cpu().numpy()

        if len(ids) > 0 and len(det_bboxes) > 0:
            track_velocities = torch.stack(
                [self.tracks[id].velocity for id in ids]).to(det_bboxes.device)
            k_step_observations = torch.stack([
                self.k_step_observation(self.tracks[id]) for id in ids
            ]).to(det_bboxes.device)
            # valid1: if the track has previous observations to estimate speed
            # valid2: if the associated observation k steps ago is a detection
            valid1 = track_velocities.sum(dim=1) != -2
            valid2 = k_step_observations.sum(dim=1) != -4
            valid = valid1 & valid2

            vel_to_match = self.vel_direction_batch(k_step_observations[:, :4],
                                                    det_bboxes[:, :4])
            track_velocities = track_velocities[:, None, :].repeat(
                1, det_bboxes.shape[0], 1)

            angle_cos = (vel_to_match * track_velocities).sum(dim=-1)
            angle_cos = torch.clamp(angle_cos, min=-1, max=1)
            angle = torch.acos(angle_cos)  # [0, pi]
            norm_angle = (angle - np.pi / 2.) / np.pi  # [-0.5, 0.5]
            valid_matrix = valid[:, None].int().repeat(1, det_bboxes.shape[0])
            # set non-valid entries 0
            valid_norm_angle = norm_angle * valid_matrix

            dists += valid_norm_angle.cpu().numpy() * self.vel_consist_weight

        # bipartite match
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col

    def last_obs(self, track):
        """extract the last associated observation."""
        for bbox in track.obs[::-1]:
            if bbox is not None:
                return bbox

    def ocr_assign_ids(self,
                       track_obs,
                       det_bboxes,
                       weight_iou_with_det_scores=False,
                       match_iou_thr=0.5):
        """association for Observation-Centric Recovery.

        As try to recover tracks from being lost whose estimated velocity is
        out- to-date, we use IoU-only matching strategy.

        Args:
            track_obs (Tensor): the list of historical associated
                detections of tracks
            det_bboxes (Tensor): of shape (N, 5), unmatched detections
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.

        Returns:
            tuple(int): The assigning ids.
        """
        # compute distance
        ious = bbox_overlaps(track_obs[:, :4], det_bboxes[:, :4])
        if weight_iou_with_det_scores:
            ious *= det_bboxes[:, 4][None]

        dists = (1 - ious).cpu().numpy()

        # bipartite match
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
        else:
            row = np.zeros(len(track_obs)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col

    def online_smooth(self, track, obj):
        """Once a track is recovered from being lost, online smooth its
        parameters to fix the error accumulated during being lost.

        NOTE: you can use different virtual trajectory generation
        strategies, we adopt the naive linear interpolation as default
        """
        last_match_bbox = self.last_obs(track)[:4]
        new_match_bbox = obj[:4]
        unmatch_len = 0
        for bbox in track.obs[::-1]:
            if bbox is None:
                unmatch_len += 1
            else:
                break
        bbox_shift_per_step = (new_match_bbox - last_match_bbox) / (
            unmatch_len + 1)
        track.mean = track.saved_attr.mean
        track.covariance = track.saved_attr.covariance
        for i in range(unmatch_len):
            virtual_bbox = last_match_bbox + (i + 1) * bbox_shift_per_step
            virtual_bbox = bbox_xyxy_to_cxcyah(virtual_bbox[None, :])
            virtual_bbox = virtual_bbox.squeeze(0).cpu().numpy()
            track.mean, track.covariance = self.kf.update(
                track.mean, track.covariance, virtual_bbox)

    @force_fp32(apply_to=('img', 'bboxes'))
    def track(self,
              img,
              img_metas,
              model,
              bboxes,
              labels,
              frame_id,
              rescale=False,
              **kwargs):
        """Tracking forward function.
        NOTE: this implementation is slightly different from the original
        OC-SORT implementation (https://github.com/noahcao/OC_SORT)that we
        do association between detections and tentative/non-tentative tracks
        independently while the original implementation combines them together.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            model (nn.Module): MOT model.
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.
        Returns:
            tuple: Tracking results.
        """
        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if self.empty or bboxes.size(0) == 0:
            valid_inds = bboxes[:, -1] > self.init_track_thr
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks
        else:
            # 0. init
            ids = torch.full((bboxes.size(0), ),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)

            # get the detection bboxes for the first association
            det_inds = bboxes[:, -1] > self.obj_score_thr
            det_bboxes = bboxes[det_inds]
            det_labels = labels[det_inds]
            det_ids = ids[det_inds]

            # 1. predict by Kalman Filter
            for id in self.confirmed_ids:
                # track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                if self.tracks[id].tracked:
                    self.tracks[id].saved_attr.mean = self.tracks[id].mean
                    self.tracks[id].saved_attr.covariance = self.tracks[
                        id].covariance
                (self.tracks[id].mean,
                 self.tracks[id].covariance) = self.kf.predict(
                     self.tracks[id].mean, self.tracks[id].covariance)

            # 2. match detections and tracks' predicted locations
            match_track_inds, raw_match_det_inds = self.ocm_assign_ids(
                self.confirmed_ids, det_bboxes,
                self.weight_iou_with_det_scores, self.match_iou_thr)
            # '-1' mean a detection box is not matched with tracklets in
            # previous frame
            valid = raw_match_det_inds > -1
            det_ids[valid] = torch.tensor(
                self.confirmed_ids)[raw_match_det_inds[valid]].to(labels)

            match_det_bboxes = det_bboxes[valid]
            match_det_labels = det_labels[valid]
            match_det_ids = det_ids[valid]
            assert (match_det_ids > -1).all()

            # unmatched tracks and detections
            unmatch_det_bboxes = det_bboxes[~valid]
            unmatch_det_labels = det_labels[~valid]
            unmatch_det_ids = det_ids[~valid]
            assert (unmatch_det_ids == -1).all()

            # 3. use unmatched detection bboxes from the first match to match
            # the unconfirmed tracks
            (tentative_match_track_inds,
             tentative_match_det_inds) = self.ocm_assign_ids(
                 self.unconfirmed_ids, unmatch_det_bboxes,
                 self.weight_iou_with_det_scores, self.match_iou_thr)
            valid = tentative_match_det_inds > -1
            unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            match_det_bboxes = torch.cat(
                (match_det_bboxes, unmatch_det_bboxes[valid]), dim=0)
            match_det_labels = torch.cat(
                (match_det_labels, unmatch_det_labels[valid]), dim=0)
            match_det_ids = torch.cat((match_det_ids, unmatch_det_ids[valid]),
                                      dim=0)
            assert (match_det_ids > -1).all()

            unmatch_det_bboxes = unmatch_det_bboxes[~valid]
            unmatch_det_labels = unmatch_det_labels[~valid]
            unmatch_det_ids = unmatch_det_ids[~valid]
            assert (unmatch_det_ids == -1).all()

            all_track_ids = [id for id, _ in self.tracks.items()]
            unmatched_track_inds = torch.tensor(
                [ind for ind in all_track_ids if ind not in match_det_ids])

            if len(unmatched_track_inds) > 0:
                # 4. still some tracks not associated yet, perform OCR
                last_observations = []
                for id in unmatched_track_inds:
                    last_box = self.last_obs(self.tracks[id.item()])
                    last_observations.append(last_box)
                last_observations = torch.stack(last_observations)

                remain_det_ids = torch.full((unmatch_det_bboxes.size(0), ),
                                            -1,
                                            dtype=labels.dtype,
                                            device=labels.device)

                _, ocr_match_det_inds = self.ocr_assign_ids(
                    last_observations, unmatch_det_bboxes,
                    self.weight_iou_with_det_scores, self.match_iou_thr)

                valid = ocr_match_det_inds > -1
                remain_det_ids[valid] = unmatched_track_inds.clone()[
                    ocr_match_det_inds[valid]].to(labels)

                ocr_match_det_bboxes = unmatch_det_bboxes[valid]
                ocr_match_det_labels = unmatch_det_labels[valid]
                ocr_match_det_ids = remain_det_ids[valid]
                assert (ocr_match_det_ids > -1).all()

                ocr_unmatch_det_bboxes = unmatch_det_bboxes[~valid]
                ocr_unmatch_det_labels = unmatch_det_labels[~valid]
                ocr_unmatch_det_ids = remain_det_ids[~valid]
                assert (ocr_unmatch_det_ids == -1).all()

                unmatch_det_bboxes = ocr_unmatch_det_bboxes
                unmatch_det_labels = ocr_unmatch_det_labels
                unmatch_det_ids = ocr_unmatch_det_ids
                match_det_bboxes = torch.cat(
                    (match_det_bboxes, ocr_match_det_bboxes), dim=0)
                match_det_labels = torch.cat(
                    (match_det_labels, ocr_match_det_labels), dim=0)
                match_det_ids = torch.cat((match_det_ids, ocr_match_det_ids),
                                          dim=0)

            # 5. summarize the track results
            for i in range(len(match_det_ids)):
                det_bbox = match_det_bboxes[i]
                track_id = match_det_ids[i].item()
                if not self.tracks[track_id].tracked:
                    # the track is lost before this step
                    self.online_smooth(self.tracks[track_id], det_bbox)

            for track_id in all_track_ids:
                if track_id not in match_det_ids:
                    self.tracks[track_id].tracked = False
                    self.tracks[track_id].obs.append(None)

            bboxes = torch.cat((match_det_bboxes, unmatch_det_bboxes), dim=0)
            labels = torch.cat((match_det_labels, unmatch_det_labels), dim=0)
            ids = torch.cat((match_det_ids, unmatch_det_ids), dim=0)

            # 6. assign new ids
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()

        self.update(ids=ids, bboxes=bboxes, labels=labels, frame_ids=frame_id)
        return bboxes, labels, ids
