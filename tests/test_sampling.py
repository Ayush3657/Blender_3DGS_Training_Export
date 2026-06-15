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

print("\nALL SAMPLING TESTS PASSED")
