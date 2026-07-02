"""End-to-end test for scripts/fuse_depth_init.py.

Builds a synthetic dataset (tilted plane, two cameras with non-trivial poses,
analytic depth EXRs + patterned PNGs + COLMAP txt model), runs the fusion
script, then verifies:
  1. every fused point lies on the plane (catches unprojection/pose errors)
  2. point colors match the pixel pattern when re-projected (catches u/v flips)

Requires: numpy, OpenEXR, Pillow  (same deps as the script).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

try:
    import OpenEXR
    from PIL import Image
except ImportError:
    print("SKIP: OpenEXR/Pillow not installed")
    sys.exit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "blender_3dgs_export"))
import colmap_io  # noqa: E402

W, H, FX, FY, CX, CY = 160, 120, 140.0, 140.0, 80.0, 60.0
PLANE_N = np.array([0.15, 0.25, 0.956])
PLANE_N = PLANE_N / np.linalg.norm(PLANE_N)
PLANE_C = 3.0                                   # plane: n . X = c


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def make_camera(R_w2c, t_w2c, name, out):
    """Analytic depth of the plane + patterned color image for one camera."""
    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dirs = np.stack([(us - CX) / FX, (vs - CY) / FY, np.ones_like(us)], axis=-1)
    # plane n.X_w = c with X_w = R^T(x_cam - t), ray x_cam = dir*s:
    #   n.R^T dir * s - n.R^T t = c  and  n.R^T v = (R n).v
    m = R_w2c @ PLANE_N
    denom = dirs @ m
    s = (PLANE_C + m @ t_w2c) / denom            # planar depth (z_cam)
    assert (s > 0.1).all(), "plane must be in front of the camera"
    depth = s.astype(np.float32)

    hdr = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    OpenEXR.File(hdr, {"depth.V": OpenEXR.Channel(depth)}).write(
        os.path.join(out, "depths", name + ".exr"))

    img = np.zeros((H, W, 3), dtype=np.uint8)   # pattern encodes pixel coords
    img[..., 0] = (np.arange(W)[None, :]) % 256
    img[..., 1] = (np.arange(H)[:, None]) % 256
    img[..., 2] = 77
    Image.fromarray(img).save(os.path.join(out, "images", name + ".png"))


def main():
    out = tempfile.mkdtemp(prefix="fusetest_")
    for sub in ("images", "depths", os.path.join("sparse", "0")):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # camera 1: identity pose; camera 2: rotated + translated
    poses = []
    R1, t1 = np.eye(3), np.zeros(3)
    c2w_R2 = rot_y(np.radians(25.0))
    C2 = np.array([0.6, -0.4, -0.5])            # camera 2 center
    R2 = c2w_R2.T                                # world-to-cam
    t2 = -R2 @ C2
    poses = [("cam1", R1, t1), ("cam2", R2, t2)]

    cams = [{"id": 1, "model": "PINHOLE", "width": W, "height": H,
             "params": [FX, FY, CX, CY]}]
    images = []
    for i, (name, R, t) in enumerate(poses):
        make_camera(R, t, name, out)
        q = mat_to_quat(R)
        images.append({"id": i + 1, "qvec": tuple(q), "tvec": tuple(t),
                       "camera_id": 1, "name": name + ".png"})
    sparse = os.path.join(out, "sparse", "0")
    colmap_io.write_cameras_txt(os.path.join(sparse, "cameras.txt"), cams)
    colmap_io.write_images_txt(os.path.join(sparse, "images.txt"), images)

    # run the fusion script
    script = os.path.join(ROOT, "scripts", "fuse_depth_init.py")
    r = subprocess.run([sys.executable, script, "--data", out,
                        "--points", "20000", "--edge-thresh", "0",
                        "--oversample", "6"],
                       capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stderr

    # read back and verify
    pts, cols = [], []
    for line in open(os.path.join(sparse, "points3D.txt")):
        if line.startswith("#"):
            continue
        t = line.split()
        pts.append([float(x) for x in t[1:4]])
        cols.append([int(x) for x in t[4:7]])
    pts = np.array(pts); cols = np.array(cols)
    assert len(pts) > 5000, f"too few points: {len(pts)}"

    # 1. on-plane check
    err = np.abs(pts @ PLANE_N - PLANE_C)
    print(f"plane residual: max {err.max():.2e}  mean {err.mean():.2e}")
    assert err.max() < 1e-3, "points not on the plane -> unprojection bug"

    # 2. color/pixel consistency: project into cam1's frame where possible
    name, R, t = poses[0]
    pc = pts @ R.T + t
    infront = pc[:, 2] > 0.1
    u = FX * pc[infront, 0] / pc[infront, 2] + CX
    v = FY * pc[infront, 1] / pc[infront, 2] + CY
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    # points sampled FROM cam1 have colors encoding their own pixel coords;
    # points from cam2 encode cam2 pixels — so only assert on the subset whose
    # color matches within quantization when reprojected (>40% must match)
    uu = np.clip(u[inside].astype(int), 0, W - 1)
    vv = np.clip(v[inside].astype(int), 0, H - 1)
    expect_r = uu % 256
    expect_g = vv % 256
    got = cols[infront][inside]
    match = (np.abs(got[:, 0] - expect_r) <= 1) & (np.abs(got[:, 1] - expect_g) <= 1)
    frac = match.mean()
    print(f"color/pixel match fraction (cam1 subset): {frac:.2f}")
    assert frac > 0.40, "colors don't map back to pixels -> u/v flip bug"
    assert (cols[:, 2] == 77).all(), "blue channel corrupted"

    print("\nALL FUSION TESTS PASSED")


if __name__ == "__main__":
    main()
