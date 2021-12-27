from .data import SupervisedFrameDataset

import torch

import csv

_TYPE_MAP = SupervisedFrameDataset.lab_enum
_INVERSE_TYPE_MAP = {v: k for k, v in _TYPE_MAP.items()}


def _find_orphans(seg_types, seg_lengths, k1, kb):

    n_1_orphans = (seg_types != 2) & (seg_lengths < k1)
    n_b_orphans = (seg_types == 2) & (seg_lengths < kb)

    return n_1_orphans + n_b_orphans


def _make_mask(tensor, idx):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[idx] = False
    return mask


class Segmentation:
    def __init__(self, scores):

        highest_scores, predicted_classes = torch.max(scores, dim=1)

        end_frames = torch.where(predicted_classes[1:] != predicted_classes[:-1])[0]
        end_frames = torch.cat(
            (end_frames, torch.tensor([predicted_classes.shape[0] - 1]))
        )
        start_frames = torch.cat(
            (torch.zeros([1], dtype=torch.int64), end_frames[:-1] + 1)
        )

        self.te = {
            "end_frames": end_frames,
            "frame_types": predicted_classes[end_frames],
            "run_lengths": torch.cat(
                (end_frames[0].unsqueeze(0) + 1, end_frames[1:] - end_frames[:-1])
            ),
            "start_frames": start_frames,
            "score_means": torch.stack(
                [
                    highest_scores[start : end + 1].mean()
                    for start, end in zip(start_frames, end_frames)
                ]
            ),
        }

    def __len__(self):
        return self.te["end_frames"].shape[0]

    def _mask_tensors(self, mask):

        self.te = {k: v[mask] for k, v in self.te.items()}

    def _update_neighbor(self, orphan_idx, neighbor_idx):

        self.te["start_frames"][neighbor_idx] = 0
        self.te["score_means"][neighbor_idx] = (
            self.te["score_means"][neighbor_idx] * self.te["run_lengths"][neighbor_idx]
            + self.te["score_means"][orphan_idx] * self.te["run_lengths"][orphan_idx]
        ) / self.te["run_lengths"][neighbor_idx] + self.te["run_lengths"][orphan_idx]
        self.te["run_lengths"][neighbor_idx] = (
            self.te["end_frames"][neighbor_idx]
            - self.te["start_frames"][neighbor_idx]
            + 1
        )

    def glue_orphans(self, real_threshold=100, blank_threshold=10):

        orphan_mask = _find_orphans(
            self.te["frame_types"],
            self.te["run_lengths"],
            real_threshold,
            blank_threshold,
        )

        while orphan_mask.sum() > 0:

            orphan_idx = torch.arange(self.te["end_frames"].shape[0])[orphan_mask]
            # Choose the least confident first.
            target_idx = orphan_idx[
                torch.argsort(self.te["score_means"][orphan_mask])[0].item()
            ]

            # If it's the first element
            if target_idx == 0:

                next_idx = target_idx + 1
                self.update_neighbor(target_idx, next_idx)

                r_mask = _make_mask(self.te["start_frames"], target_idx.item())
                self._mask_tensors(r_mask)

                orphan_mask = _find_orphans(
                    self.te["frame_types"],
                    self.te["run_lengths"],
                    real_threshold,
                    blank_threshold,
                )

            # If it's the last element
            elif target_idx == self.te["start_frames"].shape[0]:

                previous_idx = target_idx - 1
                self.update_neighbor(target_idx, previous_idx)

                r_mask = _make_mask(self.te["start_frames"], target_idx.item())
                self._mask_tensors(r_mask)

                orphan_mask = _find_orphans(
                    self.te["frame_types"],
                    self.te["run_lengths"],
                    real_threshold,
                    blank_threshold,
                )

            # If it's an in-between element
            else:

                previous_idx = target_idx - 1
                next_idx = target_idx + 1

                # We want to take the class from the larger neighbor.
                if (
                    self.te["run_lengths"][previous_idx]
                    > self.te["run_lengths"][next_idx]
                ):

                    self._update_neighbor(target_idx, previous_idx)

                else:
                    self._update_neighbor(target_idx, next_idx)

                r_mask = _make_mask(self.te["start_frames"], [target_idx.item()])
                self._mask_tensors(r_mask)

                orphan_mask = _find_orphans(
                    self.te["frame_types"],
                    self.te["run_lengths"],
                    real_threshold,
                    blank_threshold,
                )

    def write_csv(self, file_path):

        rows = [
            (sf.item(), _INVERSE_TYPE_MAP[tp.item()])
            for sf, tp in zip(self.te["start_frames"], self.te["frame_types"])
        ]

        with open(file_path, "w", newline="") as f:
            cw = csv.writer(f, delimiter=",")
            for r in rows:
                cw.writerow(r)
