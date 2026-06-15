"""Validate transforms.json output: structure, identity/known matrices, FOV,
per-frame intrinsic overrides, and the camera-to-world convention."""
import os, sys, json, tempfile, math
sys.path.insert(0, os.path.join(os.getcwd(), "blender_3dgs_export"))
import transforms_io

W, H, FX, FY, CX, CY = 1920, 1080, 1662.768, 1662.768, 960.0, 540.0

# Frame 0: camera looking down -Z at origin from z=5 (identity-ish c2w).
# matrix_world columns: right=+X, up=+Y, back=+Z, pos.
c2w_a = [[1,0,0, 1.5],[0,1,0,-2.0],[0,0,1, 3.25],[0,0,0,1]]
# Frame 1: different focal length (per-frame override expected).
c2w_b = [[0,-1,0, 0.0],[1,0,0, 0.0],[0,0,1, 2.0],[0,0,0,1]]

frames = [
    {'file_path':'images/GSCam_000.png','transform_matrix':c2w_a,
     'w':W,'h':H,'fl_x':FX,'fl_y':FY,'cx':CX,'cy':CY},
    {'file_path':'images/GSCam_001.png','transform_matrix':c2w_b,
     'w':W,'h':H,'fl_x':800.0,'fl_y':800.0,'cx':CX,'cy':CY},   # different fl
]

d = tempfile.mkdtemp()
p = os.path.join(d, "transforms.json")
transforms_io.write_transforms(p, frames, ply_path="sparse/0/points3D.ply")
data = json.load(open(p))

# top-level intrinsics from first frame
assert data["camera_model"] == "OPENCV"
assert data["w"]==W and data["h"]==H
assert abs(data["fl_x"]-FX)<1e-9 and abs(data["fl_y"]-FY)<1e-9
assert data["cx"]==CX and data["cy"]==CY
assert data["k1"]==0.0 and data["p1"]==0.0
assert data["ply_file_path"]=="sparse/0/points3D.ply"

# camera_angle_x must match 2*atan(w/(2 fl_x))
exp_ax = 2.0*math.atan(W/(2.0*FX))
assert abs(data["camera_angle_x"]-exp_ax)<1e-12, (data["camera_angle_x"], exp_ax)
# sanity: ~ 1.08 rad (~62 deg) for this focal
assert abs(math.degrees(data["camera_angle_x"]) - 60.0) < 0.5

assert len(data["frames"])==2
f0, f1 = data["frames"]
# frame 0 uses global intrinsics -> no per-frame keys
assert "fl_x" not in f0, "frame0 should not override intrinsics"
assert f0["file_path"]=="images/GSCam_000.png"
assert f0["transform_matrix"]==c2w_a
# frame 1 has different fl -> per-frame override present
assert abs(f1["fl_x"]-800.0)<1e-9 and f1["w"]==W
assert f1["transform_matrix"]==c2w_b
print("transforms.json structure + per-frame override: OK")

# --- convention check: c2w columns are camera basis vectors (OpenGL) ---
# For c2w_a (identity rotation): camera right=+X, up=+Y, back=+Z, look=-Z.
import numpy as np
M = np.array(c2w_a, dtype=float)
right, up, back, pos = M[:3,0], M[:3,1], M[:3,2], M[:3,3]
assert np.allclose(right,[1,0,0]) and np.allclose(up,[0,1,0]) and np.allclose(back,[0,0,1])
look = -back
assert np.allclose(look,[0,0,-1]), "camera must look down -Z (OpenGL/Blender)"
assert np.allclose(pos,[1.5,-2.0,3.25])
print("camera-to-world OpenGL convention (look = -Z): OK")

# --- cross-check vs COLMAP path: c2w_gl = c2w_cv @ diag(1,-1,-1) ---
# Build world-to-cam OpenCV from c2w_a, then verify the relationship holds.
R_BCAM2CV = np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
w2c_cv = np.zeros((4,4)); 
Rcw = M[:3,:3]; pos3 = M[:3,3]
# world->blendercam = inv(M)
w2b = np.linalg.inv(M)
R_w2c = R_BCAM2CV @ w2b[:3,:3]; t = R_BCAM2CV @ w2b[:3,3]
# COLMAP camera center
C = -R_w2c.T @ t
assert np.allclose(C, pos3), "COLMAP center must equal transforms.json position"
# c2w_cv = inv([R|t]) ; then c2w_cv @ diag(1,-1,-1) should equal M[:3,:3]
c2w_cv_R = R_w2c.T
recon = c2w_cv_R @ np.diag([1,-1,-1])
assert np.allclose(recon, Rcw, atol=1e-9), "c2w_gl = c2w_cv @ diag(1,-1,-1) mismatch"
print("COLMAP <-> transforms.json consistency: OK")

print("\nALL TRANSFORMS TESTS PASSED")
