"""Writers for the COLMAP sparse model: cameras, images, points3D.

Supports both the text (.txt) and binary (.bin) variants. The binary format
follows COLMAP's `scripts/python/read_write_model.py` exactly.

Data shapes expected by the writers
-----------------------------------
cameras : list of dicts
    {'id': int, 'model': 'PINHOLE', 'width': int, 'height': int,
     'params': [fx, fy, cx, cy]}
images : list of dicts
    {'id': int, 'qvec': (qw,qx,qy,qz), 'tvec': (tx,ty,tz),
     'camera_id': int, 'name': str}
points : xyz (N,3) float array, rgb (N,3) uint8 array
"""

import struct
import numpy as np

# COLMAP camera model name -> model id
CAMERA_MODEL_IDS = {
    'SIMPLE_PINHOLE': 0,
    'PINHOLE': 1,
    'SIMPLE_RADIAL': 2,
    'RADIAL': 3,
    'OPENCV': 4,
}


def _f(x):
    """Round-trippable float formatting for text files."""
    return repr(float(x))


# --------------------------------------------------------------------------- #
# Text writers
# --------------------------------------------------------------------------- #
def write_cameras_txt(path, cameras):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for c in cameras:
            params = " ".join(_f(p) for p in c['params'])
            f.write(f"{c['id']} {c['model']} {c['width']} {c['height']} {params}\n")


def write_images_txt(path, images):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
        for im in images:
            qw, qx, qy, qz = im['qvec']
            tx, ty, tz = im['tvec']
            f.write(f"{im['id']} {_f(qw)} {_f(qx)} {_f(qy)} {_f(qz)} "
                    f"{_f(tx)} {_f(ty)} {_f(tz)} {im['camera_id']} {im['name']}\n")
            f.write("\n")  # empty POINTS2D line (no 2D-3D correspondences needed)


def write_points3D_txt(path, xyz, rgb):
    n = len(xyz)
    xyz = np.asarray(xyz)
    rgb = np.asarray(rgb)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {n}, mean track length: 0\n")
        lines = []
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            lines.append(f"{i + 1} {_f(x)} {_f(y)} {_f(z)} {int(r)} {int(g)} {int(b)} 0")
        if lines:
            f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Binary writers
# --------------------------------------------------------------------------- #
def write_cameras_bin(path, cameras):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for c in cameras:
            model_id = CAMERA_MODEL_IDS[c['model']]
            # camera_id (uint32), model_id (int32), width (uint64), height (uint64)
            f.write(struct.pack('<IiQQ', c['id'], model_id, c['width'], c['height']))
            params = c['params']
            f.write(struct.pack('<' + 'd' * len(params), *[float(p) for p in params]))


def write_images_bin(path, images):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(images)))
        for im in images:
            f.write(struct.pack('<I', im['id']))
            f.write(struct.pack('<4d', *[float(v) for v in im['qvec']]))
            f.write(struct.pack('<3d', *[float(v) for v in im['tvec']]))
            f.write(struct.pack('<I', im['camera_id']))
            f.write(im['name'].encode('utf-8') + b'\x00')  # null-terminated name
            f.write(struct.pack('<Q', 0))                   # num_points2D = 0


# Packed (unaligned) record layout for points3D.bin:
#   id(uint64) x,y,z(float64) r,g,b(uint8) error(float64) track_length(uint64)
_POINT3D_DTYPE = np.dtype([
    ('id', '<u8'),
    ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ('err', '<f8'),
    ('trk', '<u8'),
])


def write_points3D_bin(path, xyz, rgb):
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    n = len(xyz)
    rec = np.zeros(n, dtype=_POINT3D_DTYPE)
    if n:
        rec['id'] = np.arange(1, n + 1, dtype=np.uint64)
        rec['x'] = xyz[:, 0]
        rec['y'] = xyz[:, 1]
        rec['z'] = xyz[:, 2]
        rec['r'] = rgb[:, 0]
        rec['g'] = rgb[:, 1]
        rec['b'] = rgb[:, 2]
        # err and trk already zero
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', n))
        f.write(rec.tobytes())


# --------------------------------------------------------------------------- #
# Convenience: write a whole sparse model in the requested format(s)
# --------------------------------------------------------------------------- #
def write_model(sparse_dir, cameras, images, xyz, rgb, fmt='BOTH'):
    """fmt in {'TXT', 'BIN', 'BOTH'}."""
    import os
    if fmt in ('TXT', 'BOTH'):
        write_cameras_txt(os.path.join(sparse_dir, 'cameras.txt'), cameras)
        write_images_txt(os.path.join(sparse_dir, 'images.txt'), images)
        write_points3D_txt(os.path.join(sparse_dir, 'points3D.txt'), xyz, rgb)
    if fmt in ('BIN', 'BOTH'):
        write_cameras_bin(os.path.join(sparse_dir, 'cameras.bin'), cameras)
        write_images_bin(os.path.join(sparse_dir, 'images.bin'), images)
        write_points3D_bin(os.path.join(sparse_dir, 'points3D.bin'), xyz, rgb)


# --------------------------------------------------------------------------- #
# Optional PLY (handy for previewing the init cloud / some pipelines want it)
# --------------------------------------------------------------------------- #
_PLY_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
])


def write_points_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    n = len(xyz)
    rec = np.zeros(n, dtype=_PLY_DTYPE)
    if n:
        rec['x'] = xyz[:, 0]
        rec['y'] = xyz[:, 1]
        rec['z'] = xyz[:, 2]
        rec['r'] = rgb[:, 0]
        rec['g'] = rgb[:, 1]
        rec['b'] = rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode('ascii')
    with open(path, 'wb') as f:
        f.write(header)
        f.write(rec.tobytes())


_PLY_ORIENTED_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
])


def write_points_ply_oriented(path, xyz, nrm, rgb):
    """Surface-aligned seed cloud: position + normal + color per point."""
    xyz = np.asarray(xyz, dtype=np.float32)
    nrm = np.asarray(nrm, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    n = len(xyz)
    rec = np.zeros(n, dtype=_PLY_ORIENTED_DTYPE)
    if n:
        rec['x'] = xyz[:, 0]; rec['y'] = xyz[:, 1]; rec['z'] = xyz[:, 2]
        rec['nx'] = nrm[:, 0]; rec['ny'] = nrm[:, 1]; rec['nz'] = nrm[:, 2]
        rec['r'] = rgb[:, 0]; rec['g'] = rgb[:, 1]; rec['b'] = rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode('ascii')
    with open(path, 'wb') as f:
        f.write(header)
        f.write(rec.tobytes())
