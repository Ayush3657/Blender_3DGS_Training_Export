"""Pure (bpy-free) sampling helpers for the walkthrough-baking workflow.

Kept dependency-free so the resampling math can be unit-tested without Blender.
"""

import numpy as np


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
