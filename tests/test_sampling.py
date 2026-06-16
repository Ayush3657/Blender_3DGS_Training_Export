"""Validate arc-length resampling and frame-step helpers (bpy-free)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "blender_3dgs_export"))
import sampling

# --- frame_step_list ---
assert sampling.frame_step_list(1, 10, 5) == [1, 6, 10], sampling.frame_step_list(1,10,5)
assert sampling.frame_step_list(1, 10, 3) == [1, 4, 7, 10]   # last always included
assert sampling.frame_step_list(10, 1, 5) == [1, 6, 10]      # handles reversed
assert sampling.frame_step_list(5, 5, 5) == [5]
print("frame_step_list: OK")

# --- even-distance: straight line, 11 pts spaced 1.0 (total length 10) ---
line = np.stack([np.arange(11.0), np.zeros(11), np.zeros(11)], axis=1)
idx = sampling.resample_indices_by_distance(line, 2.5)
# targets 0,2.5,5,7.5,10 -> first frame at/after each cumulative distance
assert idx[0] == 0 and idx[-1] == 10, idx
# spacing between selected positions ~2.5 (within one segment)
sel = line[idx][:, 0]
gaps = np.diff(sel)
assert gaps.min() >= 2.0 and gaps.max() <= 3.0, (sel, gaps)
print(f"even-distance on straight line: indices={idx}  positions={sel.tolist()}: OK")

# --- variable speed: dwell (many close pts) then a fast sprint ---
# 20 pts clustered within 0.2 units, then a jump segment of length 8 in 2 steps.
dwell = np.stack([np.linspace(0, 0.2, 20), np.zeros(20), np.zeros(20)], axis=1)
sprint = np.array([[4.0, 0, 0], [8.0, 0, 0]])
path = np.vstack([dwell, sprint])
# frame-step every 5 frames would put 4 samples in the dwell, 0 in the sprint.
fstep = sampling.frame_step_list(0, len(path) - 1, 5)
dwell_hits_fs = sum(1 for i in fstep if i < 20)
# distance sampling at 1.0 should instead spread across the sprint, not clump.
idist = sampling.resample_indices_by_distance(path, 1.0)
dwell_hits_dist = sum(1 for i in idist if i < 20)
assert dwell_hits_dist <= 2, ("distance sampling should not clump in the dwell", idist)
assert max(idist) == len(path) - 1
print(f"variable-speed: frame-step put {dwell_hits_fs} samples in the dwell, "
      f"distance put {dwell_hits_dist}: OK")

# --- degenerate cases ---
assert sampling.resample_indices_by_distance([], 1.0) == []
assert sampling.resample_indices_by_distance([[1, 2, 3]], 1.0) == [0]
# stationary camera (no movement) -> single sample
assert sampling.resample_indices_by_distance(np.zeros((10, 3)), 0.5) == [0]
# never exceeds the number of frames even with tiny spacing
big = sampling.resample_indices_by_distance(line, 0.01)
assert len(big) <= len(line) and max(big) == 10
print("degenerate cases (empty / single / stationary / tiny spacing): OK")

# --- motion sampler: STAND STILL AND PAN (the user's case) ---
import math as _m
N = 20
still = np.zeros((N, 3))                                  # camera never moves
angs = np.linspace(0.0, _m.pi, N)                         # view pans 0..180 deg
fwd = np.stack([np.cos(angs), np.sin(angs), np.zeros(N)], axis=1)
# distance disabled (0), rotation every 30 deg -> must produce interior samples
idx = sampling.resample_indices_by_motion(still, fwd, 0.0, _m.radians(30))
assert idx[0] == 0 and idx[-1] == N - 1
assert len(idx) >= 5, ("rotation in place should still create cameras", idx)
# pure distance sampling would give only one sample here:
assert sampling.resample_indices_by_distance(still, 0.5) == [0]
print(f"stand-still-and-pan: distance->1 sample, motion(30deg)->{len(idx)} samples: OK")

# --- motion sampler: move straight, no rotation -> distance trigger only ---
line2 = np.stack([np.arange(11.0), np.zeros(11), np.zeros(11)], axis=1)
fwd_const = np.tile([0.0, -1.0, 0.0], (11, 1))
idxm = sampling.resample_indices_by_motion(line2, fwd_const, 2.5, _m.radians(30))
assert idxm[0] == 0 and idxm[-1] == 10
gapsm = np.diff(line2[idxm][:, 0])
assert gapsm.max() <= 3.0, (idxm, gapsm)
print(f"motion sampler, translation only: indices={idxm}: OK")

# --- both triggers disabled -> endpoints only; empty/single ---
assert sampling.resample_indices_by_motion(line2, fwd_const, 0.0, 0.0) == [0, 10]
assert sampling.resample_indices_by_motion([], [], 0.5, 0.5) == []
assert sampling.resample_indices_by_motion([[0, 0, 0]], [[0, 0, -1]], 0.5, 0.5) == [0]
print("motion sampler degenerate cases: OK")

# --- region_bounds_from_centers (camera-region limiting) ---
# Cameras in a small room near origin; bounds must exclude far-away geometry.
cam_centers = np.array([[-0.7, 0.26, 0.33], [2.19, 5.73, 3.0], [0.9, 2.6, 1.6]])
lo, hi = sampling.region_bounds_from_centers(cam_centers, 2.0)
# extent ~ [2.89, 5.47, 2.67]; pad = max(2.0, extent) -> expands by >= extent
assert (lo < cam_centers.min(0)).all() and (hi > cam_centers.max(0)).all()
# a far point like the bad export (224534, 65032, -59076) must be OUTSIDE
far = np.array([224534.0, 65032.0, -59076.0])
assert not (np.all(far >= lo) and np.all(far <= hi)), "far junk must be excluded"
# the room itself (a few metres around the cameras) must be INSIDE
room_pt = np.array([1.0, 3.0, 0.0])  # floor under the walk
assert np.all(room_pt >= lo) and np.all(room_pt <= hi), (lo, hi)
# padding floor: tiny camera spread still expands by at least `padding`
lo2, hi2 = sampling.region_bounds_from_centers(np.array([[0., 0, 0], [0.1, 0, 0]]), 2.0)
assert lo2[1] <= -2.0 and hi2[1] >= 2.0
assert sampling.region_bounds_from_centers([], 2.0) is None
print(f"region_bounds_from_centers: room in, far junk out  (lo={np.round(lo,2)}, hi={np.round(hi,2)}): OK")

print("\nALL SAMPLING TESTS PASSED")
