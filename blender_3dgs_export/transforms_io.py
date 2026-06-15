"""Writer for transforms.json (NeRF / Nerfstudio / instant-ngp format).

Coordinate convention
---------------------
Unlike COLMAP (world-to-camera, OpenCV axes), transforms.json stores the
*camera-to-world* matrix in the **OpenGL / Blender** convention:
    +X right, +Y up, -Z forward (camera looks down -Z).

That is exactly Blender's native camera frame, so `transform_matrix` is just the
camera's `matrix_world` dumped row-major — the same thing the original NeRF
Blender data-generation script does. No axis conversion is applied here.

Frame dict shape expected by write_transforms()
    {'file_path': 'images/GSCam_000.png',
     'transform_matrix': [[..4x4 row-major..]],   # = camera.matrix_world
     'w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'}
"""

import json
import math


def _intr_key(d):
    return (d['w'], d['h'], round(d['fl_x'], 6), round(d['fl_y'], 6),
            round(d['cx'], 6), round(d['cy'], 6))


def write_transforms(path, frames, ply_path=None, camera_model="OPENCV", aabb_scale=16):
    """Write a transforms.json. Global intrinsics come from the first frame;
    any frame whose intrinsics differ gets a per-frame override (supported by
    Nerfstudio / instant-ngp)."""
    if not frames:
        return

    g = frames[0]
    base_key = _intr_key(g)

    data = {
        "camera_model": camera_model,
        "w": g['w'],
        "h": g['h'],
        "fl_x": g['fl_x'],
        "fl_y": g['fl_y'],
        "cx": g['cx'],
        "cy": g['cy'],
        # No lens distortion (perfect pinhole renders).
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0,
        # Legacy fields for older NeRF loaders that key off FOV.
        "camera_angle_x": 2.0 * math.atan(g['w'] / (2.0 * g['fl_x'])),
        "camera_angle_y": 2.0 * math.atan(g['h'] / (2.0 * g['fl_y'])),
        # instant-ngp scene-bounds hint (ignored by loaders that don't use it).
        "aabb_scale": aabb_scale,
    }
    if ply_path:
        data["ply_file_path"] = ply_path  # used by Nerfstudio splatfacto for init

    out_frames = []
    for fr in frames:
        frame = {
            "file_path": fr['file_path'],
            "transform_matrix": fr['transform_matrix'],
        }
        if _intr_key(fr) != base_key:
            frame.update({
                "w": fr['w'], "h": fr['h'],
                "fl_x": fr['fl_x'], "fl_y": fr['fl_y'],
                "cx": fr['cx'], "cy": fr['cy'],
            })
        out_frames.append(frame)
    data["frames"] = out_frames

    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2)
