#!/usr/bin/env python3
"""Fuse exported GT depth maps + rendered images into a dense, LIT initial
point cloud for 3DGS training ("trainer as refiner").

Reads a dataset produced by the 3DGS Training Export Blender addon:

    <data>/
      images/   <name>.png        rendered beauty (colors, sRGB)
      depths/   <stem>.exr        metric planar-Z depth (channel "depth.V")
      sparse/0/ cameras.txt|bin, images.txt|bin   COLMAP model (PINHOLE)

and REPLACES sparse/0/points3D.{bin,txt,ply} with a fused cloud where every
point sits exactly on rendered geometry and carries the *rendered, lit* color
of that surface (unlike the addon's surface-sampled cloud, which uses flat
material base colors). Originals are backed up to *.bak on first run.

Everything stays in the COLMAP model's coordinate frame (up-axis already baked
in by the addon), so the output is consistent with the poses by construction.

Usage:
    python fuse_depth_init.py --data D:/path/to/dataset [--points 2000000]
        [--max-depth 1000] [--edge-thresh 0.10] [--camera-stride 1]
        [--voxel 0] [--oversample 4] [--no-backup]

Dependencies:  pip install numpy OpenEXR Pillow
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

try:
    import OpenEXR
except ImportError:
    sys.exit("Missing dependency: pip install OpenEXR")
try:
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency: pip install Pillow")

# Reuse the addon's (bpy-free) COLMAP writers.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "blender_3dgs_export"))
import colmap_io  # noqa: E402


# --------------------------------------------------------------------------- #
# COLMAP model readers (PINHOLE model, txt preferred, bin fallback)
# --------------------------------------------------------------------------- #
def read_cameras(sparse_dir):
    txt = os.path.join(sparse_dir, "cameras.txt")
    if os.path.exists(txt):
        cams = {}
        for line in open(txt, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            t = line.split()
            cams[int(t[0])] = {"model": t[1], "w": int(t[2]), "h": int(t[3]),
                               "params": [float(x) for x in t[4:]]}
        return cams
    binp = os.path.join(sparse_dir, "cameras.bin")
    cams = {}
    with open(binp, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model_id, w, h = struct.unpack("<IiQQ", f.read(24))
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4)
            params = struct.unpack("<" + "d" * nparams, f.read(8 * nparams))
            cams[cid] = {"model": "PINHOLE", "w": w, "h": h, "params": list(params)}
    return cams


def read_images(sparse_dir):
    txt = os.path.join(sparse_dir, "images.txt")
    imgs = []
    if os.path.exists(txt):
        lines = [l for l in open(txt, encoding="utf-8") if not l.startswith("#")]
        for i in range(0, len(lines) - 1, 2):   # pose line + points2D line
            t = lines[i].split()
            if len(t) < 10:
                continue
            imgs.append({"qvec": [float(x) for x in t[1:5]],
                         "tvec": [float(x) for x in t[5:8]],
                         "camera_id": int(t[8]), "name": t[9]})
        return imgs
    binp = os.path.join(sparse_dir, "images.bin")
    with open(binp, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", f.read(4))
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n2d)
            imgs.append({"qvec": list(qvec), "tvec": list(tvec),
                         "camera_id": cam_id, "name": name.decode()})
    return imgs


def quat_to_R(q):
    """(qw,qx,qy,qz), COLMAP convention -> 3x3 world-to-camera rotation."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# --------------------------------------------------------------------------- #
# EXR depth loading (handles Blender's layered channel naming, e.g. "depth.V")
# --------------------------------------------------------------------------- #
def load_depth_exr(path):
    f = OpenEXR.File(path)
    chans = f.channels()
    key = None
    for k in chans:                       # prefer *.V, then V/Y/Z/R, then first
        if k.endswith(".V") or k in ("V", "Y", "Z", "R"):
            key = k
            break
    if key is None:
        key = next(iter(chans))
    ch = chans[key]
    arr = np.asarray(ch.pixels if hasattr(ch, "pixels") else ch, dtype=np.float32)
    if arr.ndim == 3:                     # multi-channel stored as (h, w, c)
        arr = arr[..., 0]
    return arr


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def fuse(args):
    data = os.path.abspath(args.data)
    sparse_dir = os.path.join(data, "sparse", "0")
    images_dir = os.path.join(data, "images")
    depths_dir = os.path.join(data, "depths")
    for p in (sparse_dir, images_dir, depths_dir):
        if not os.path.isdir(p):
            sys.exit(f"Not found: {p}")

    cams = read_cameras(sparse_dir)
    imgs = read_images(sparse_dir)
    imgs.sort(key=lambda im: im["name"])
    if args.camera_stride > 1:
        imgs = imgs[::args.camera_stride]
    print(f"[fuse] {len(imgs)} views  |  target {args.points:,} points")

    per_cam = max(64, int(args.oversample * args.points / max(1, len(imgs))))
    rng = np.random.default_rng(args.seed)

    xyz_list, rgb_list = [], []
    used = skipped = 0
    t0 = time.time()
    for i, im in enumerate(imgs):
        stem = os.path.splitext(im["name"])[0]
        dpath = os.path.join(depths_dir, stem + ".exr")
        ipath = os.path.join(images_dir, im["name"])
        if not (os.path.exists(dpath) and os.path.exists(ipath)):
            skipped += 1
            continue
        cam = cams[im["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4]

        depth = load_depth_exr(dpath)
        h, w = depth.shape

        valid = np.isfinite(depth) & (depth > 0.0) & (depth < args.max_depth)
        # Drop depth-discontinuity pixels (silhouettes): the HQ beauty is
        # antialiased there, so their colors mix foreground/background.
        if args.edge_thresh > 0:
            rel = args.edge_thresh * depth
            edge = np.zeros_like(valid)
            edge[:, :-1] |= np.abs(np.diff(depth, axis=1)) > rel[:, :-1]
            edge[:-1, :] |= np.abs(np.diff(depth, axis=0)) > rel[:-1, :]
            valid &= ~edge

        ys, xs = np.nonzero(valid)
        if xs.size == 0:
            skipped += 1
            continue
        if xs.size > per_cam:
            sel = rng.choice(xs.size, size=per_cam, replace=False)
            xs, ys = xs[sel], ys[sel]

        z = depth[ys, xs].astype(np.float64)
        u = xs.astype(np.float64) + 0.5
        v = ys.astype(np.float64) + 0.5
        pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)

        R = quat_to_R(im["qvec"])
        t = np.asarray(im["tvec"], dtype=np.float64)
        pw = (pc - t) @ R                       # = R^T (pc - t)

        img = np.asarray(Image.open(ipath).convert("RGB"), dtype=np.uint8)
        if img.shape[0] != h or img.shape[1] != w:
            skipped += 1
            continue
        col = img[ys, xs]

        xyz_list.append(pw.astype(np.float32))
        rgb_list.append(col)
        used += 1
        if (i + 1) % 200 == 0 or i + 1 == len(imgs):
            print(f"[fuse] {i + 1}/{len(imgs)} views  "
                  f"({sum(len(a) for a in xyz_list):,} candidates, "
                  f"{time.time() - t0:.0f}s)")

    if not xyz_list:
        sys.exit("No points produced — check depths/ matches images/ names.")
    xyz = np.concatenate(xyz_list).astype(np.float64)
    rgb = np.concatenate(rgb_list)
    print(f"[fuse] views used {used}, skipped {skipped}  |  "
          f"candidates {len(xyz):,}")

    # ---- voxel downsample to an even surface density -----------------------
    lo, hi = xyz.min(0), xyz.max(0)
    print(f"[fuse] bbox  x[{lo[0]:.2f},{hi[0]:.2f}] "
          f"y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]")
    if args.voxel > 0:
        voxel = args.voxel
    else:
        # auto: iterate to land the unique-voxel count near the target
        # (surface points: unique count scales ~ 1/voxel^2)
        vol = float(np.prod(np.maximum(hi - lo, 1e-6)))
        voxel = (vol / max(args.points, 1)) ** (1.0 / 3.0)
        for _ in range(10):
            keys = np.floor((xyz - lo) / voxel).astype(np.int64)
            uniq = len(np.unique(keys.view([('', np.int64)] * 3)))
            ratio = uniq / args.points
            if 0.9 <= ratio <= 1.15:
                break
            voxel *= ratio ** 0.5
    keys = np.floor((xyz - lo) / voxel).astype(np.int64)
    kview = keys.view([('', np.int64)] * 3).ravel()
    order = rng.permutation(len(xyz))          # random keep-one-per-voxel
    _, first = np.unique(kview[order], return_index=True)
    keep = order[first]
    if len(keep) > args.points:
        keep = rng.choice(keep, size=args.points, replace=False)
    xyz, rgb = xyz[keep], rgb[keep]
    print(f"[fuse] voxel {voxel * 100:.1f} cm  ->  {len(xyz):,} points")

    # ---- backup + write -----------------------------------------------------
    for name in ("points3D.bin", "points3D.txt", "points3D.ply"):
        p = os.path.join(sparse_dir, name)
        if args.backup and os.path.exists(p) and not os.path.exists(p + ".bak"):
            os.replace(p, p + ".bak")
    colmap_io.write_points3D_bin(os.path.join(sparse_dir, "points3D.bin"), xyz, rgb)
    colmap_io.write_points3D_txt(os.path.join(sparse_dir, "points3D.txt"), xyz, rgb)
    # Always refresh the .ply too: some trainers (LichtFeld) cache one from a
    # previous run and would silently keep training on the old cloud.
    colmap_io.write_points_ply(os.path.join(sparse_dir, "points3D.ply"), xyz, rgb)
    print(f"[fuse] wrote points3D.bin/.txt/.ply -> {sparse_dir}")
    print("[fuse] done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root (contains images/, depths/, sparse/0/)")
    ap.add_argument("--points", type=int, default=2_000_000, help="target point count (default 2M)")
    ap.add_argument("--max-depth", type=float, default=1000.0, help="ignore depths beyond this (m)")
    ap.add_argument("--edge-thresh", type=float, default=0.10,
                    help="relative depth-gradient cutoff for silhouette pixels (0 = off)")
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel size in metres (0 = auto)")
    ap.add_argument("--oversample", type=float, default=4.0, help="candidate factor before downsample")
    ap.add_argument("--camera-stride", type=int, default=1, help="use every Nth camera (quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-backup", dest="backup", action="store_false",
                    help="overwrite points3D files without .bak backups")
    fuse(ap.parse_args())


if __name__ == "__main__":
    main()
