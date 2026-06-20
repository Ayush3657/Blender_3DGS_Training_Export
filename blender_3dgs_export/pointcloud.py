"""Initial point-cloud generation for COLMAP / 3DGS.

Three strategies:
  * SURFACE : area-weighted sampling of mesh surfaces, colored from each
              material's Principled base color. Robust, fast, view-independent.
              Best default for synthetic archviz interiors.
  * DEPTH   : back-project rendered depth (Z) maps into colored world points.
              Densest and exactly matches what the cameras saw. Experimental.
  * RANDOM  : uniform points in the scene bounding box. Cheap fallback.

All points are returned in *Blender world coordinates* so they stay consistent
with the exported camera poses (see camera_utils for why).
"""

import bpy
import numpy as np


# --------------------------------------------------------------------------- #
# Color helpers
# --------------------------------------------------------------------------- #
def _lin2srgb(c):
    """Vectorized linear -> sRGB (display) transform, input/output in [0, 1]."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _principled_base_color(mat):
    """Best-effort linear RGB for a material's base color."""
    if mat is None:
        return (0.8, 0.8, 0.8)
    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                c = node.inputs['Base Color'].default_value
                return (c[0], c[1], c[2])
        for node in mat.node_tree.nodes:
            if node.type in {'BSDF_DIFFUSE', 'EMISSION'}:
                c = node.inputs['Color'].default_value
                return (c[0], c[1], c[2])
    return tuple(mat.diffuse_color[:3])


def _empty():
    return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Mesh-surface sampling
# --------------------------------------------------------------------------- #
def sample_surface(context, n_points, objects, rng=None, bounds=None, return_normals=False):
    """Area-weighted surface sampling across the given mesh objects.

    If `bounds` (lo, hi) is given, only triangles whose centroid falls inside the
    box are sampled (and any stray sampled point outside the box is dropped), so
    far/huge geometry elsewhere in the scene can't dominate the cloud.

    If `return_normals` is True, returns (xyz, rgb, normals) with per-point
    world-space surface normals (face normals via the inverse-transpose normal
    matrix); otherwise returns (xyz, rgb).
    """
    if rng is None:
        rng = np.random.default_rng()
    deps = context.evaluated_depsgraph_get()
    lo = hi = None
    if bounds is not None:
        lo = np.asarray(bounds[0], dtype=np.float64)
        hi = np.asarray(bounds[1], dtype=np.float64)

    v0_list, v1_list, v2_list, col_list, nrm_list = [], [], [], [], []

    for obj in objects:
        obj_eval = obj.evaluated_get(deps)
        try:
            me = obj_eval.to_mesh()
        except Exception:
            continue
        if me is None:
            continue
        try:
            # calc_loop_triangles() populates me.loop_triangles; on versions where
            # triangulation is implicit this is a harmless no-op.
            try:
                me.calc_loop_triangles()
            except Exception:
                pass
            nv = len(me.vertices)
            nt = len(me.loop_triangles)
            if nv == 0 or nt == 0:
                continue

            co = np.empty(nv * 3, dtype=np.float64)
            me.vertices.foreach_get('co', co)
            co = co.reshape(-1, 3)

            tri = np.empty(nt * 3, dtype=np.int64)
            me.loop_triangles.foreach_get('vertices', tri)
            tri = tri.reshape(-1, 3)

            mat_idx = np.empty(nt, dtype=np.int64)
            me.loop_triangles.foreach_get('material_index', mat_idx)

            # Transform vertices to world space.
            M = np.array(obj.matrix_world, dtype=np.float64)
            co_w = co @ M[:3, :3].T + M[:3, 3]

            t0 = co_w[tri[:, 0]]
            t1 = co_w[tri[:, 1]]
            t2 = co_w[tri[:, 2]]

            slots = obj.material_slots
            if len(slots) > 0:
                cols = np.array([_principled_base_color(s.material) for s in slots],
                                dtype=np.float64)
            else:
                cols = np.array([[0.8, 0.8, 0.8]], dtype=np.float64)
            mat_idx = np.clip(mat_idx, 0, len(cols) - 1)
            tcol = cols[mat_idx]

            tnrm = None
            if return_normals:
                tn = np.empty(nt * 3, dtype=np.float64)
                me.loop_triangles.foreach_get('normal', tn)
                tn = tn.reshape(-1, 3)
                # World-space normals use the inverse-transpose of the 3x3 (handles
                # non-uniform scale); fall back to the plain 3x3 if singular.
                try:
                    nmat = np.linalg.inv(M[:3, :3]).T
                except np.linalg.LinAlgError:
                    nmat = M[:3, :3]
                tnrm = tn @ nmat.T
                norms = np.linalg.norm(tnrm, axis=1, keepdims=True)
                tnrm = tnrm / np.where(norms == 0, 1.0, norms)

            # Keep only triangles whose centroid is inside the camera region.
            if lo is not None:
                cent = (t0 + t1 + t2) / 3.0
                m = np.all((cent >= lo) & (cent <= hi), axis=1)
                if not m.any():
                    continue
                t0, t1, t2, tcol = t0[m], t1[m], t2[m], tcol[m]
                if tnrm is not None:
                    tnrm = tnrm[m]

            v0_list.append(t0)
            v1_list.append(t1)
            v2_list.append(t2)
            col_list.append(tcol)
            if return_normals:
                nrm_list.append(tnrm)
        finally:
            obj_eval.to_mesh_clear()

    if not v0_list:
        return (np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros((0, 3))) \
            if return_normals else _empty()

    v0 = np.concatenate(v0_list)
    v1 = np.concatenate(v1_list)
    v2 = np.concatenate(v2_list)
    tri_col = np.concatenate(col_list)
    tri_nrm = np.concatenate(nrm_list) if return_normals else None

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = areas.sum()
    if total <= 0 or n_points <= 0:
        return (np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros((0, 3))) \
            if return_normals else _empty()

    probs = areas / total
    idx = rng.choice(len(areas), size=int(n_points), p=probs)

    # Uniform barycentric sampling inside each chosen triangle.
    u1 = rng.random(int(n_points))
    u2 = rng.random(int(n_points))
    su1 = np.sqrt(u1)
    b0 = (1.0 - su1)[:, None]
    b1 = (su1 * (1.0 - u2))[:, None]
    b2 = (su1 * u2)[:, None]
    pts = b0 * v0[idx] + b1 * v1[idx] + b2 * v2[idx]
    col = tri_col[idx]
    nrm = tri_nrm[idx] if return_normals else None

    # Drop any stray samples outside the region (e.g. from huge straddling tris).
    if lo is not None:
        m = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts, col = pts[m], col[m]
        if nrm is not None:
            nrm = nrm[m]

    rgb = (_lin2srgb(col) * 255.0).round().clip(0, 255).astype(np.uint8)
    if return_normals:
        return pts, rgb, nrm
    return pts, rgb


# --------------------------------------------------------------------------- #
# Random fallback
# --------------------------------------------------------------------------- #
def sample_random(context, n_points, objects, rng=None, bounds=None):
    if rng is None:
        rng = np.random.default_rng()
    if bounds is not None:
        # Sample uniformly in the camera region directly.
        mins = np.asarray(bounds[0], dtype=np.float64)
        maxs = np.asarray(bounds[1], dtype=np.float64)
        if np.allclose(mins, maxs):
            maxs = mins + 1.0
        pts = rng.uniform(mins, maxs, size=(int(n_points), 3))
        rgb = np.full((int(n_points), 3), 180, dtype=np.uint8)
        return pts, rgb
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for obj in objects:
        M = np.array(obj.matrix_world, dtype=np.float64)
        for corner in obj.bound_box:
            w = M[:3, :3] @ np.array(corner, dtype=np.float64) + M[:3, 3]
            mins = np.minimum(mins, w)
            maxs = np.maximum(maxs, w)
    if not np.all(np.isfinite(mins)):
        return _empty()
    if np.allclose(mins, maxs):
        maxs = mins + 1.0
    pts = rng.uniform(mins, maxs, size=(int(n_points), 3))
    rgb = np.full((int(n_points), 3), 180, dtype=np.uint8)
    return pts, rgb


# --------------------------------------------------------------------------- #
# Depth back-projection (experimental)
# --------------------------------------------------------------------------- #
def _load_image_array(path):
    """Load an image and return (h, w, channels) float32 array in TOP-origin
    order, read as raw (non-color-managed) values."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        try:
            img.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
        w, h = img.size
        ch = img.channels
        buf = np.empty(w * h * ch, dtype=np.float32)
        img.pixels.foreach_get(buf)
        buf = buf.reshape(h, w, ch)
        return buf[::-1]  # flip from Blender bottom-origin to top-origin
    finally:
        bpy.data.images.remove(img)


def points_from_depth(cam_records, n_points, rng=None, far_clip=1.0e7):
    """Back-project per-camera depth maps into a single colored world cloud.

    cam_records : list of dicts with keys
        depth_path, image_path, fx, fy, cx, cy, R (3x3 np), t (3 np)
    n_points : total budget across all cameras
    """
    if rng is None:
        rng = np.random.default_rng()
    if not cam_records:
        return _empty()

    per_cam = max(1, int(n_points) // len(cam_records))
    xyz_list, rgb_list = [], []

    for rec in cam_records:
        try:
            depth = _load_image_array(rec['depth_path'])[..., 0]
            color = _load_image_array(rec['image_path'])
        except Exception:
            continue

        h, w = depth.shape
        valid = np.isfinite(depth) & (depth > 0.0) & (depth < far_clip)
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            continue
        if len(xs) > per_cam:
            sel = rng.choice(len(xs), size=per_cam, replace=False)
            xs, ys = xs[sel], ys[sel]

        z = depth[ys, xs].astype(np.float64)
        u = xs.astype(np.float64) + 0.5
        v = ys.astype(np.float64) + 0.5

        # Pixel -> OpenCV camera coordinates.
        xc = (u - rec['cx']) / rec['fx'] * z
        yc = (v - rec['cy']) / rec['fy'] * z
        pc = np.stack([xc, yc, z], axis=1)               # (N, 3)

        # Camera -> world:  X_world = R^T (X_cam - t)   ->  row form: (pc - t) @ R
        pw = (pc - rec['t'][None, :]) @ rec['R']
        xyz_list.append(pw)

        col = color[ys, xs, :3]
        rgb = (np.clip(col, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        rgb_list.append(rgb)

    if not xyz_list:
        return _empty()
    return np.concatenate(xyz_list), np.concatenate(rgb_list)
