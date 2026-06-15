"""Camera math: convert a Blender camera into COLMAP intrinsics and extrinsics.

COLMAP / OpenCV camera convention:
    +X right, +Y down, +Z forward (camera looks down +Z).
Blender camera convention:
    +X right, +Y up,  -Z forward (camera looks down -Z).

The poses we export are *world-to-camera* (the rotation R and translation t such
that  X_cam = R * X_world + t ), stored as a quaternion (qw, qx, qy, qz) + (tx, ty, tz),
which is exactly what COLMAP's images file expects.

Important: the 3D points we export live in *Blender world coordinates*. The OpenCV
flip is baked entirely into R, so points and poses stay consistent (the camera
center -R^T t equals the Blender camera world location).
"""

import numpy as np
from mathutils import Matrix

# Rotation that maps a point from the Blender camera frame to the OpenCV camera frame.
# Flip Y and Z (it is its own inverse).
R_BCAM2CV = Matrix(((1.0, 0.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (0.0, 0.0, -1.0)))


def compute_intrinsics(cam_data, render):
    """Return (width, height, fx, fy, cx, cy) in pixels for a PINHOLE camera.

    Uses the *scaled* resolution (resolution_percentage), which is the size of the
    image Blender actually writes to disk.
    """
    scale = render.resolution_percentage / 100.0
    width = max(1, int(round(render.resolution_x * scale)))
    height = max(1, int(round(render.resolution_y * scale)))

    pixel_aspect = render.pixel_aspect_x / render.pixel_aspect_y  # normally 1.0

    f_mm = cam_data.lens          # focal length in mm (kept in sync regardless of lens_unit)
    sensor_w = cam_data.sensor_width
    sensor_h = cam_data.sensor_height

    fit = cam_data.sensor_fit
    if fit == 'AUTO':
        # Blender fits the sensor to the larger pixel-aspect-corrected dimension.
        fit = 'HORIZONTAL' if (width >= height * pixel_aspect) else 'VERTICAL'

    if fit == 'HORIZONTAL':
        # sensor_width maps to the full image width
        fx = f_mm * width / sensor_w
        fy = fx * pixel_aspect
    else:  # VERTICAL: sensor_height maps to the full image height
        fy = f_mm * height / sensor_h
        fx = fy / pixel_aspect

    # Principal point, including optical lens shift (shift_* default to 0).
    # Blender expresses shift as a fraction of the larger image dimension.
    big = max(width, height)
    cx = width * 0.5 + cam_data.shift_x * big
    cy = height * 0.5 - cam_data.shift_y * big

    return width, height, fx, fy, cx, cy


def get_extrinsics(cam_obj):
    """Return world-to-camera pose of a Blender camera object in COLMAP convention.

    Returns:
        qvec : (qw, qx, qy, qz) tuple, normalized
        tvec : (tx, ty, tz) tuple
        R_np : 3x3 numpy array (world-to-camera rotation, OpenCV)
        t_np : 3 numpy array (world-to-camera translation, OpenCV)
    """
    # matrix_world is camera-to-world (Blender frame). Invert to get world-to-bcam.
    world2bcam = cam_obj.matrix_world.inverted()
    R_w2bc = world2bcam.to_3x3()
    t_w2bc = world2bcam.to_translation()

    # Convert from Blender camera frame to OpenCV camera frame.
    R = R_BCAM2CV @ R_w2bc
    t = R_BCAM2CV @ t_w2bc

    q = R.to_quaternion()
    q.normalize()

    qvec = (q.w, q.x, q.y, q.z)
    tvec = (t.x, t.y, t.z)
    R_np = np.array(R, dtype=np.float64)        # mathutils 3x3 -> (3, 3)
    t_np = np.array(tvec, dtype=np.float64)
    return qvec, tvec, R_np, t_np
