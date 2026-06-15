"""Pure (bpy-free) sampling helpers for the walkthrough-baking workflow.

Kept dependency-free so the resampling math can be unit-tested without Blender.
"""

import math

import numpy as np


def resample_indices_by_motion(positions, forwards, distance_spacing, angle_spacing_rad):
    """Keep a frame whenever EITHER distance travelled OR view rotation since the
    last kept frame crosses a threshold.

    This captures "stand still and pan to face a different wall" (rotation) just
    as well as walking (translation) — pure distance sampling would skip the pan
    because no distance accumulates.

    positions         : (N, 3) per-frame camera positions (frame order)
    forwards          : (N, 3) per-frame camera view directions (camera -Z, world)
    distance_spacing  : metres between samples; <= 0 disables the distance trigger
    angle_spacing_rad : radians between samples; <= 0 disables the rotation trigger

    Returns a sorted list of unique indices; first and last frames always kept.
    """
    pos = np.asarray(positions, dtype=float)
    fwd = np.asarray(forwards, dtype=float)
    n = len(pos)
    if n == 0:
        return []
    if n == 1:
        return [0]

    use_d = distance_spacing is not None and distance_spacing > 0
    use_a = angle_spacing_rad is not None and angle_spacing_rad > 0
    if not use_d and not use_a:
        return [0, n - 1]

    norms = np.linalg.norm(fwd, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    fwd_n = fwd / norms

    keep = [0]
    acc_d = 0.0
    acc_a = 0.0
    for i in range(1, n):
        acc_d += float(np.linalg.norm(pos[i] - pos[i - 1]))
        dot = float(np.clip(np.dot(fwd_n[i], fwd_n[i - 1]), -1.0, 1.0))
        acc_a += math.acos(dot)
        if (use_d and acc_d >= distance_spacing) or (use_a and acc_a >= angle_spacing_rad):
            keep.append(i)
            acc_d = 0.0
            acc_a = 0.0
    if keep[-1] != n - 1:
        keep.append(n - 1)
    return keep


def resample_indices_by_distance(positions, spacing):
    """Pick frame indices ~evenly spaced by distance travelled along a path.

    positions : (N, 3) array of camera positions, in frame order.
    spacing   : desired distance between consecutive samples (world units).

    Returns a sorted list of unique indices into `positions`. The first and last
    frames are always included. Sampling by arc length (rather than by frame
    number) keeps coverage even when the camera moves at a variable speed.
    """
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if n == 0:
        return []
    if n == 1 or spacing <= 0:
        return [0]

    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])   # cumulative distance per frame
    total = float(cum[-1])
    if total <= 0.0:
        return [0]

    targets = np.arange(0.0, total + 1e-9, spacing)
    idx = np.searchsorted(cum, targets)             # first frame at/after each target
    idx = np.clip(idx, 0, n - 1)

    return sorted(set(idx.tolist()) | {0, n - 1})


def frame_step_list(frame_start, frame_end, step):
    """Frames from start to end (inclusive) every `step`, always including the end."""
    if frame_end < frame_start:
        frame_start, frame_end = frame_end, frame_start
    step = max(1, int(step))
    frames = list(range(frame_start, frame_end + 1, step))
    if frames and frames[-1] != frame_end:
        frames.append(frame_end)
    return frames
