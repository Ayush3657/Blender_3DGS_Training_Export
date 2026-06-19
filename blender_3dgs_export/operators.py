"""Operators: add a camera array, and render + export a COLMAP dataset."""

import os
import re
import glob
import math
import shutil

import bpy
import numpy as np
from bpy.props import (
    EnumProperty, IntProperty, FloatProperty, BoolProperty, StringProperty,
    FloatVectorProperty,
)
from bpy.types import Operator
from mathutils import Vector

from . import camera_utils, colmap_io, pointcloud, transforms_io, sampling

CAM_COLLECTION_NAME = "3DGS_Cameras"
IMAGE_EXT = {'PNG': '.png', 'JPEG': '.jpg'}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _sanitize(name):
    name = re.sub(r'[^A-Za-z0-9_.\-]', '_', name)
    return name or "cam"


def _collection_in_scene(scene, coll):
    def walk(c):
        if c == coll:
            return True
        return any(walk(ch) for ch in c.children)
    return walk(scene.collection)


def _get_or_create_collection(context, props):
    coll = props.camera_collection
    if coll is None:
        coll = bpy.data.collections.get(CAM_COLLECTION_NAME)
        if coll is None:
            coll = bpy.data.collections.new(CAM_COLLECTION_NAME)
        props.camera_collection = coll
    # Make sure it's linked into the scene so its cameras are renderable.
    if not _collection_in_scene(context.scene, coll):
        try:
            context.scene.collection.children.link(coll)
        except Exception:
            pass
    return coll


def _cameras_in_collection(coll):
    if coll is None:
        return []
    cams = [o for o in coll.all_objects if o.type == 'CAMERA']
    cams.sort(key=lambda o: o.name)
    return cams


def _resolve_source_camera(context):
    """The camera to sample: the active object if it's a camera, else the scene camera."""
    obj = context.active_object
    if obj is not None and obj.type == 'CAMERA':
        return obj
    cam = context.scene.camera
    if cam is not None and cam.type == 'CAMERA':
        return cam
    return None


# --------------------------------------------------------------------------- #
# Add Camera Array
# --------------------------------------------------------------------------- #
class GS_OT_add_camera_array(Operator):
    bl_idname = "gs_export.add_camera_array"
    bl_label = "Add Camera Array"
    bl_description = "Create an array of cameras (in the 3DGS camera collection) that you can then reposition"
    bl_options = {'REGISTER', 'UNDO'}

    pattern: EnumProperty(
        name="Pattern",
        items=[
            ('GRID', "Grid (box)", "3D grid of camera positions"),
            ('LINE', "Line", "Cameras along a straight line"),
            ('ARC', "Arc / Orbit", "Cameras on a horizontal arc around the center"),
            ('DOME', "Dome", "Cameras on a hemisphere above the center"),
        ],
        default='GRID',
    )
    center: FloatVectorProperty(
        name="Center", subtype='TRANSLATION', default=(0.0, 0.0, 1.5),
        description="Origin of the array (and look-at target if enabled)",
    )
    look_at_center: BoolProperty(
        name="Aim at Center", default=True,
        description="Orient cameras toward the center point (orbit/dome aim inward)",
    )

    count_x: IntProperty(name="Count X", default=3, min=1, max=64)
    count_y: IntProperty(name="Count Y", default=3, min=1, max=64)
    count_z: IntProperty(name="Count Z", default=1, min=1, max=64)
    spacing: FloatProperty(name="Spacing", default=1.0, min=0.0, subtype='DISTANCE')

    count: IntProperty(name="Count", default=12, min=1, max=512)
    radius: FloatProperty(name="Radius", default=4.0, min=0.0, subtype='DISTANCE')
    arc_degrees: FloatProperty(name="Arc (degrees)", default=360.0, min=0.0, max=360.0)
    dome_rings: IntProperty(name="Dome Rings", default=3, min=1, max=32)

    focal_length: FloatProperty(name="Focal Length (mm)", default=24.0, min=1.0)
    clip_start: FloatProperty(name="Clip Start", default=0.01, min=0.0001, subtype='DISTANCE')
    clip_end: FloatProperty(name="Clip End", default=1000.0, min=0.1, subtype='DISTANCE')

    def _new_camera(self, context, coll, name, location, target=None):
        cam_data = bpy.data.cameras.new(name)
        cam_data.lens = self.focal_length
        cam_data.clip_start = self.clip_start
        cam_data.clip_end = self.clip_end
        obj = bpy.data.objects.new(name, cam_data)
        obj.location = location
        if target is not None and self.look_at_center:
            direction = (Vector(target) - Vector(location))
            if direction.length > 1e-6:
                obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        coll.objects.link(obj)
        return obj

    def execute(self, context):
        props = context.scene.gs_export
        coll = _get_or_create_collection(context, props)
        c = Vector(self.center)
        target = self.center
        made = 0

        if self.pattern == 'GRID':
            ox = (self.count_x - 1) * self.spacing * 0.5
            oy = (self.count_y - 1) * self.spacing * 0.5
            oz = (self.count_z - 1) * self.spacing * 0.5
            for ix in range(self.count_x):
                for iy in range(self.count_y):
                    for iz in range(self.count_z):
                        loc = (c.x + ix * self.spacing - ox,
                               c.y + iy * self.spacing - oy,
                               c.z + iz * self.spacing - oz)
                        self._new_camera(context, coll, f"GSCam_{made:03d}", loc, target)
                        made += 1

        elif self.pattern == 'LINE':
            length = (self.count - 1) * self.spacing
            for i in range(self.count):
                loc = (c.x + i * self.spacing - length * 0.5, c.y, c.z)
                self._new_camera(context, coll, f"GSCam_{made:03d}", loc, target)
                made += 1

        elif self.pattern == 'ARC':
            span = math.radians(self.arc_degrees)
            full = abs(self.arc_degrees - 360.0) < 1e-3
            denom = self.count if full else max(1, self.count - 1)
            for i in range(self.count):
                a = (span * i / denom) if self.count > 1 else 0.0
                loc = (c.x + self.radius * math.cos(a),
                       c.y + self.radius * math.sin(a),
                       c.z)
                self._new_camera(context, coll, f"GSCam_{made:03d}", loc, target)
                made += 1

        elif self.pattern == 'DOME':
            for ring in range(self.dome_rings):
                elev = (math.pi * 0.5) * (ring + 0.5) / self.dome_rings  # 0..90 deg
                per_ring = max(1, int(round(self.count / self.dome_rings)))
                r = self.radius * math.cos(elev)
                z = self.radius * math.sin(elev)
                for j in range(per_ring):
                    a = 2.0 * math.pi * j / per_ring
                    loc = (c.x + r * math.cos(a), c.y + r * math.sin(a), c.z + z)
                    self._new_camera(context, coll, f"GSCam_{made:03d}", loc, target)
                    made += 1

        self.report({'INFO'}, f"Created {made} camera(s) in '{coll.name}'")
        return {'FINISHED'}

    def invoke(self, context, event):
        # Default the array center to the 3D cursor for convenience.
        self.center = tuple(context.scene.cursor.location)
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "pattern")
        layout.prop(self, "center")
        layout.prop(self, "look_at_center")
        layout.separator()
        if self.pattern == 'GRID':
            row = layout.row(align=True)
            row.prop(self, "count_x"); row.prop(self, "count_y"); row.prop(self, "count_z")
            layout.prop(self, "spacing")
        elif self.pattern == 'LINE':
            layout.prop(self, "count")
            layout.prop(self, "spacing")
        elif self.pattern == 'ARC':
            layout.prop(self, "count")
            layout.prop(self, "radius")
            layout.prop(self, "arc_degrees")
        elif self.pattern == 'DOME':
            layout.prop(self, "count")
            layout.prop(self, "radius")
            layout.prop(self, "dome_rings")
        layout.separator()
        layout.prop(self, "focal_length")
        col = layout.column(align=True)
        col.prop(self, "clip_start")
        col.prop(self, "clip_end")


# --------------------------------------------------------------------------- #
# Walkthrough capture: prepare + bake cameras from an animated camera
# --------------------------------------------------------------------------- #
class GS_OT_prepare_walkthrough(Operator):
    bl_idname = "gs_export.prepare_walkthrough"
    bl_label = "Prepare Walkthrough"
    bl_description = ("Make the active camera the scene camera, lock it to the viewport, "
                     "and enable auto-keyframing so you can 'drive' a path with Walk navigation")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _resolve_source_camera(context) is not None

    def execute(self, context):
        scene = context.scene
        cam = _resolve_source_camera(context)
        if cam is None:
            self.report({'ERROR'}, "Add a camera (or set a scene camera) first")
            return {'CANCELLED'}

        scene.camera = cam
        locked = 0
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and hasattr(space, 'lock_camera'):
                        space.lock_camera = True
                        locked += 1
        scene.tool_settings.use_keyframe_insert_auto = True

        self.report(
            {'INFO'},
            "Locked '%s' to view + auto-key ON. Walk nav (Shift+`): fly to a spot, "
            "press I > Location & Rotation to drop a waypoint, advance the frame, repeat. "
            "Then run 'Bake Cameras from Animation'." % cam.name)
        return {'FINISHED'}


class GS_OT_bake_cameras_from_anim(Operator):
    bl_idname = "gs_export.bake_cameras_from_anim"
    bl_label = "Bake Cameras from Animation"
    bl_description = ("Sample the active/scene camera over a frame range and create static "
                      "cameras (matching pose + lens) in the export collection")
    bl_options = {'REGISTER', 'UNDO'}

    frame_start: IntProperty(name="Start Frame", default=1)
    frame_end: IntProperty(name="End Frame", default=250)
    sampling_mode: EnumProperty(
        name="Spacing",
        items=[
            ('EVEN_COVERAGE', "Even Coverage",
             "Drop a camera whenever you've moved far enough OR turned far enough — "
             "captures standing still and panning, and avoids clumping at variable speed"),
            ('FRAME_STEP', "Every N Frames", "Sample one camera every N frames"),
        ],
        default='EVEN_COVERAGE',
    )
    frame_step: IntProperty(name="Frame Step", default=5, min=1, max=1000)
    spacing: FloatProperty(
        name="Distance", default=0.5, min=0.0, soft_max=10.0, subtype='DISTANCE',
        description="New camera after this much travel (0 = ignore distance, use rotation only)")
    angle_spacing: FloatProperty(
        name="Rotation (°)", default=30.0, min=0.0, soft_max=180.0,
        description="New camera after the view turns this many degrees — captures "
                    "panning in place (0 = ignore rotation, use distance only)")
    match_intrinsics: BoolProperty(
        name="Copy Focal Length", default=True,
        description="Copy the source camera's (possibly animated) lens at each sampled frame")
    clear_collection: BoolProperty(
        name="Clear Collection First", default=False,
        description="Remove existing cameras from the export collection before baking")

    @classmethod
    def poll(cls, context):
        return _resolve_source_camera(context) is not None

    def invoke(self, context, event):
        self.frame_start = context.scene.frame_start
        self.frame_end = context.scene.frame_end
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        src = _resolve_source_camera(context)
        layout.label(text="Source: %s" % (src.name if src else "—"), icon='CON_CAMERASOLVER')
        row = layout.row(align=True)
        row.prop(self, "frame_start")
        row.prop(self, "frame_end")
        layout.prop(self, "sampling_mode")
        if self.sampling_mode == 'FRAME_STEP':
            layout.prop(self, "frame_step")
        else:
            col = layout.column(align=True)
            col.prop(self, "spacing")
            col.prop(self, "angle_spacing")
        layout.prop(self, "match_intrinsics")
        layout.prop(self, "clear_collection")

    def execute(self, context):
        scene = context.scene
        src = _resolve_source_camera(context)
        if src is None:
            self.report({'ERROR'}, "No active or scene camera to sample")
            return {'CANCELLED'}

        props = scene.gs_export
        coll = _get_or_create_collection(context, props)

        if self.clear_collection:
            for o in list(coll.objects):
                if o.type == 'CAMERA':
                    bpy.data.objects.remove(o, do_unlink=True)

        frames = sampling.frame_step_list(self.frame_start, self.frame_end, 1)
        orig_frame = scene.frame_current
        made = 0
        try:
            if self.sampling_mode == 'FRAME_STEP':
                sel_frames = sampling.frame_step_list(self.frame_start, self.frame_end,
                                                      self.frame_step)
            else:
                positions, forwards = [], []
                for f in frames:
                    scene.frame_set(f)
                    dg = context.evaluated_depsgraph_get()
                    mw = src.evaluated_get(dg).matrix_world
                    loc = mw.translation
                    fwd = (mw.to_3x3() @ Vector((0.0, 0.0, -1.0)))  # camera view dir
                    positions.append((loc.x, loc.y, loc.z))
                    forwards.append((fwd.x, fwd.y, fwd.z))
                idx = sampling.resample_indices_by_motion(
                    positions, forwards, self.spacing, math.radians(self.angle_spacing))
                sel_frames = [frames[i] for i in idx]

            for f in sel_frames:
                scene.frame_set(f)
                dg = context.evaluated_depsgraph_get()
                ev = src.evaluated_get(dg)
                matrix = ev.matrix_world.copy()

                name = "Scan_%04d" % f
                cam_data = bpy.data.cameras.new(name)
                sd = src.data
                cam_data.sensor_width = sd.sensor_width
                cam_data.sensor_height = sd.sensor_height
                cam_data.sensor_fit = sd.sensor_fit
                cam_data.shift_x = sd.shift_x
                cam_data.shift_y = sd.shift_y
                cam_data.clip_start = sd.clip_start
                cam_data.clip_end = sd.clip_end
                cam_data.lens = ev.data.lens if self.match_intrinsics else sd.lens

                obj = bpy.data.objects.new(name, cam_data)
                coll.objects.link(obj)
                obj.matrix_world = matrix
                made += 1
        finally:
            scene.frame_set(orig_frame)

        self.report({'INFO'}, "Baked %d camera(s) from '%s' into '%s'"
                    % (made, src.name, coll.name))
        return {'FINISHED'}


# --------------------------------------------------------------------------- #
# Detect & remove cameras whose view is blocked by very close geometry
# --------------------------------------------------------------------------- #
class GS_OT_cull_clipping_cameras(Operator):
    bl_idname = "gs_export.cull_clipping_cameras"
    bl_label = "Cull Clipping Cameras"
    bl_description = ("Detect cameras whose view is blocked by geometry right in front "
                      "(buried in furniture, poking through a wall) and select or delete them")
    bl_options = {'REGISTER', 'UNDO'}

    clip_distance: FloatProperty(
        name="Clip Distance", default=0.2, min=0.001, soft_max=2.0, subtype='DISTANCE',
        description="A view ray hitting geometry closer than this counts as 'blocked'")
    block_fraction: FloatProperty(
        name="Blocked Fraction", default=0.5, min=0.05, max=1.0, subtype='FACTOR',
        description="Flag the camera if at least this fraction of its view rays are blocked")
    grid: IntProperty(
        name="Ray Grid", default=7, min=2, max=21,
        description="Rays are cast in a grid×grid pattern across each camera's frustum")
    delete: BoolProperty(
        name="Delete (otherwise just Select)", default=False,
        description="Delete the flagged cameras. Off = only select them so you can review first")

    @classmethod
    def poll(cls, context):
        return bool(_cameras_in_collection(context.scene.gs_export.camera_collection))

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=320)

    def _is_clipped(self, scene, deps, cam_obj):
        cam = cam_obj.data
        mw = cam_obj.matrix_world
        origin = mw.translation
        tr, br, bl, tl = cam.view_frame(scene=scene)   # frustum corners, camera space
        n = max(2, self.grid)
        total = blocked = 0
        for i in range(n):
            s = i / (n - 1)
            for j in range(n):
                t = j / (n - 1)
                p_local = bl.lerp(br, s).lerp(tl.lerp(tr, s), t)
                direction = (mw @ p_local) - origin
                if direction.length == 0:
                    continue
                direction = direction.normalized()
                hit, loc, _, _, _, _ = scene.ray_cast(deps, origin, direction)
                total += 1
                if hit and (loc - origin).length < self.clip_distance:
                    blocked += 1
        return total > 0 and (blocked / total) >= self.block_fraction

    def execute(self, context):
        scene = context.scene
        deps = context.evaluated_depsgraph_get()
        cams = _cameras_in_collection(scene.gs_export.camera_collection)
        flagged = [c for c in cams if self._is_clipped(scene, deps, c)]

        if self.delete:
            for c in flagged:
                bpy.data.objects.remove(c, do_unlink=True)
            self.report({'INFO'}, "Deleted %d clipped camera(s) of %d"
                        % (len(flagged), len(cams)))
        else:
            for o in list(context.selected_objects):
                o.select_set(False)
            for c in flagged:
                c.select_set(True)
            if flagged:
                context.view_layer.objects.active = flagged[0]
            self.report({'INFO'}, "Selected %d clipped camera(s) of %d (review, then delete)"
                        % (len(flagged), len(cams)))
        return {'FINISHED'}


# --------------------------------------------------------------------------- #
# Render + export
# --------------------------------------------------------------------------- #
class GS_OT_render_export(Operator):
    bl_idname = "gs_export.render_export"
    bl_label = "Render & Export Dataset"
    bl_description = "Render every camera in the collection and write a COLMAP dataset"
    bl_options = {'REGISTER'}

    render: BoolProperty(default=True, options={'HIDDEN'})
    maps_only: BoolProperty(default=False, options={'HIDDEN'})

    def execute(self, context):
        scene = context.scene
        props = scene.gs_export

        out = bpy.path.abspath(props.output_dir)
        if not out:
            self.report({'ERROR'}, "Set an output folder first")
            return {'CANCELLED'}
        if not os.path.isabs(out):
            self.report({'ERROR'}, "Output path is relative ('//...'). Save your .blend "
                                   "file first, or set an absolute output folder")
            return {'CANCELLED'}

        cams = _cameras_in_collection(props.camera_collection)
        if not cams:
            self.report({'ERROR'}, "No cameras found. Assign a camera collection or use 'Add Camera Array'")
            return {'CANCELLED'}

        self.out = out
        self.images_dir = os.path.join(out, "images")
        self.sparse_dir = os.path.join(out, "sparse", "0")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.sparse_dir, exist_ok=True)

        self.scene = scene
        self.props = props
        self.cams = cams
        self.do_render = self.render
        self.depth_mode = (props.pc_mode == 'DEPTH') and self.do_render
        # Transparency requires an alpha channel -> force PNG (JPEG can't store it).
        self.image_format_eff = 'PNG' if props.transparent_bg else props.image_format
        self.restore = {}
        self.pass_state = None
        self.frame = scene.frame_current

        if self.maps_only:
            # Render ONLY the geometry passes (fast, 1 sample) for the existing
            # cameras, without re-rendering or overwriting the colour images.
            if not (props.export_depth or props.export_normal or props.export_albedo):
                self.report({'ERROR'}, "Enable at least one Ground-Truth Map first")
                return {'CANCELLED'}
            self.want_depth = props.export_depth
            self.want_normal = props.export_normal
            self.want_albedo = props.export_albedo
            self.keep_depth = True
        else:
            # Ground-truth maps written alongside the beauty (extra render passes).
            self.want_depth = self.do_render and (props.export_depth or self.depth_mode)
            self.want_normal = self.do_render and props.export_normal
            self.want_albedo = self.do_render and props.export_albedo
            self.keep_depth = props.export_depth  # depth only feeding the cloud is transient

        try:
            if self.do_render:
                self.restore = self._apply_render_settings(scene, props, cams)
            if self.want_depth or self.want_normal or self.want_albedo:
                self.pass_state = self._setup_passes(scene, context.view_layer)
            self._precompute(context)
        except Exception as exc:                       # noqa: BLE001
            self._cleanup(context)
            self.report({'ERROR'}, "Setup failed: %s" % exc)
            return {'CANCELLED'}

        # Export-only (no render): write immediately, synchronously.
        if not self.do_render:
            self._write_outputs(context, len(self.images_out))
            self._cleanup(context)
            return self._report_done()

        # Headless / no window: fall back to a blocking render loop.
        if context.window is None:
            return self._run_blocking(context)

        # Interactive: render modally so the render view shows live and the UI
        # stays responsive (Esc cancels). Each frame is launched with
        # 'INVOKE_DEFAULT' so it runs as a job rather than blocking the main thread.
        self.index = 0
        self.rendering = False
        self.stop = False
        self._timer = None
        self._add_handlers()
        self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        self._set_status(context, "3DGS: starting render…")
        return {'RUNNING_MODAL'}

    # --- precompute per-camera metadata (independent of rendering) -----------
    def _precompute(self, context):
        scene, props = self.scene, self.props
        ext = IMAGE_EXT[self.image_format_eff]
        self.cameras_out, self.images_out, self.frames_out = [], [], []
        self.cam_records, self.render_items = [], []
        intr_to_id, used_names = {}, set()

        for i, cam in enumerate(self.cams):
            w, h, fx, fy, cx, cy = camera_utils.compute_intrinsics(cam.data, scene.render)
            key = (w, h, round(fx, 4), round(fy, 4), round(cx, 4), round(cy, 4))
            if key not in intr_to_id:
                cam_id = len(self.cameras_out) + 1
                intr_to_id[key] = cam_id
                self.cameras_out.append({'id': cam_id, 'model': 'PINHOLE',
                                         'width': w, 'height': h,
                                         'params': [fx, fy, cx, cy]})
            cam_id = intr_to_id[key]

            stem = _sanitize(cam.name)
            base, n = stem, 1
            while stem in used_names:
                stem = f"{base}_{n}"
                n += 1
            used_names.add(stem)
            image_name = stem + ext

            self.render_items.append({'stem': stem,
                                      'filepath': os.path.join(self.images_dir, stem)})
            qvec, tvec, R_np, t_np = camera_utils.get_extrinsics(cam)
            self.images_out.append({'id': i + 1, 'qvec': qvec, 'tvec': tvec,
                                    'camera_id': cam_id, 'name': image_name})
            self.frames_out.append({
                'file_path': "images/" + image_name,
                'transform_matrix': [[float(v) for v in row] for row in cam.matrix_world],
                'w': w, 'h': h, 'fl_x': fx, 'fl_y': fy, 'cx': cx, 'cy': cy})
            self.cam_records.append({
                # The File Output node may append a frame suffix — resolve by glob later.
                'depth_glob': (os.path.join(self.out, "depths", stem + "_*.exr")
                               if self.want_depth else None),
                'depth_path': None,
                'image_path': os.path.join(self.images_dir, image_name),
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'R': R_np, 't': t_np})

    # --- modal driver --------------------------------------------------------
    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            self.stop = True
        if event.type == 'TIMER':
            if (self.stop or self.index >= len(self.cams)) and not self.rendering:
                return self._end(context, cancelled=self.stop)
            if not self.rendering and self.index < len(self.cams):
                self._render_next(context)
        return {'PASS_THROUGH'}

    def _render_next(self, context):
        item = self.render_items[self.index]
        self.scene.camera = self.cams[self.index]
        self.scene.render.filepath = item['filepath']
        self._set_pass_paths(item['stem'])
        self._set_status(context, "3DGS: rendering %d/%d — %s   (Esc to stop)"
                         % (self.index + 1, len(self.cams), item['stem']))
        print("[3DGS] Rendering %d/%d: %s" % (self.index + 1, len(self.cams), item['stem']))
        bpy.ops.render.render('INVOKE_DEFAULT', write_still=not self.maps_only)

    def _set_pass_paths(self, stem):
        """Name this camera's pass files (directory is fixed per node at setup)."""
        if not self.pass_state:
            return
        for fo in self.pass_state.get('fouts', []):
            fo['node'].file_name = stem + "_"

    # render handlers (bound methods; receive (scene, depsgraph) on modern Blender)
    def _on_render_pre(self, *args):
        self.rendering = True

    def _on_render_complete(self, *args):
        self.index += 1
        self.rendering = False

    def _on_render_cancel(self, *args):
        self.rendering = False
        self.stop = True

    def _add_handlers(self):
        # Insert at the FRONT so our state updates run before any third-party
        # render handler (e.g. BlenderKit) that might raise on this Blender version
        # and otherwise interrupt the chain before our finalization.
        h = bpy.app.handlers
        h.render_pre.insert(0, self._on_render_pre)
        h.render_complete.insert(0, self._on_render_complete)
        h.render_cancel.insert(0, self._on_render_cancel)
        self._handlers_added = True

    def _remove_handlers(self):
        if not getattr(self, '_handlers_added', False):
            return
        h = bpy.app.handlers
        for lst, fn in ((h.render_pre, self._on_render_pre),
                        (h.render_complete, self._on_render_complete),
                        (h.render_cancel, self._on_render_cancel)):
            if fn in lst:
                lst.remove(fn)
        self._handlers_added = False

    def _end(self, context, cancelled):
        rendered = self.index
        if getattr(self, '_timer', None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._remove_handlers()
        try:
            self._write_outputs(context, rendered)
        finally:
            self._cleanup(context)
        self._set_status(context, None)
        if cancelled:
            self.report({'WARNING'}, "Cancelled — wrote partial dataset (%d/%d views)"
                        % (rendered, len(self.cams)))
            return {'CANCELLED'}
        return self._report_done()

    # --- blocking fallback (headless only) -----------------------------------
    def _run_blocking(self, context):
        try:
            for i, item in enumerate(self.render_items):
                self.scene.camera = self.cams[i]
                self.scene.render.filepath = item['filepath']
                self._set_pass_paths(item['stem'])
                print("[3DGS] Rendering %d/%d: %s" % (i + 1, len(self.cams), item['stem']))
                bpy.ops.render.render(write_still=not self.maps_only)
            self._write_outputs(context, len(self.render_items))
        finally:
            self._cleanup(context)
        return self._report_done()

    # --- outputs + cleanup ---------------------------------------------------
    def _write_outputs(self, context, rendered):
        if self.maps_only:
            # Only the GT map EXRs were produced (by the compositor); nothing else.
            self._counts = (rendered, 0, 0)
            return
        images = self.images_out[:rendered]
        frames = self.frames_out[:rendered]
        records = self.cam_records[:rendered]
        xyz, rgb = self._build_point_cloud(context, self.props, records)
        print("[3DGS] Point cloud: %d points (%s)" % (len(xyz), self.props.pc_mode))

        # Up-axis conversion: rotate points + camera poses together (sampling and
        # bounds were done in Blender's Z-up frame; this only reorients the output).
        R_up = camera_utils.up_axis_matrix(self.props.up_axis)
        if R_up is not None:
            if len(xyz):
                xyz = np.asarray(xyz, dtype=np.float64) @ R_up.T
            for im, rec in zip(images, records):
                Rp, tp = camera_utils.rotate_world_to_cam(rec['R'], rec['t'], R_up)
                im['qvec'] = camera_utils.matrix_to_qvec(Rp)
                im['tvec'] = (float(tp[0]), float(tp[1]), float(tp[2]))
            R_up4 = np.eye(4)
            R_up4[:3, :3] = R_up
            for fr in frames:
                m = np.array(fr['transform_matrix'], dtype=np.float64)
                fr['transform_matrix'] = (R_up4 @ m).tolist()

        used_ids = {im['camera_id'] for im in images}
        cams_out = [c for c in self.cameras_out if c['id'] in used_ids] or self.cameras_out
        colmap_io.write_model(self.sparse_dir, cams_out, images, xyz, rgb,
                              fmt=self.props.colmap_format)
        if self.props.write_ply:
            colmap_io.write_points_ply(os.path.join(self.sparse_dir, "points3D.ply"), xyz, rgb)
        if self.props.write_transforms_json and frames:
            ply_ref = "sparse/0/points3D.ply" if self.props.write_ply else None
            transforms_io.write_transforms(os.path.join(self.out, "transforms.json"),
                                           frames, ply_path=ply_ref)
            print("[3DGS] Wrote transforms.json (%d frames)" % len(frames))
        self._counts = (len(images), len(cams_out), len(xyz))

    def _cleanup(self, context):
        if self.pass_state is not None:
            self._teardown_passes(self.scene, context.view_layer, self.pass_state)
            self.pass_state = None
        # Depth maps written only to seed the point cloud (not requested as an
        # output) are transient — remove them.
        if getattr(self, 'want_depth', False) and not getattr(self, 'keep_depth', False):
            d = os.path.join(self.out, "depths")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        if self.restore:
            self._restore_render_settings(self.scene, self.restore)
            self.restore = {}

    def _report_done(self):
        v, c, p = getattr(self, '_counts', (0, 0, 0))
        if self.maps_only:
            msg = "Wrote ground-truth maps for %d view(s) -> %s" % (v, self.out)
        else:
            msg = "Exported %d views, %d intrinsic(s), %d points -> %s" % (v, c, p, self.out)
        self.report({'INFO'}, msg)
        print("[3DGS] %s" % msg)
        return {'FINISHED'}

    @staticmethod
    def _set_status(context, text):
        try:
            context.workspace.status_text_set(text)
        except Exception:
            pass

    # --- render settings snapshot / restore ---------------------------------
    def _apply_render_settings(self, scene, props, cams):
        r = scene.render
        snap = {
            'filepath': r.filepath,
            'file_format': r.image_settings.file_format,
            'color_mode': r.image_settings.color_mode,
            'color_depth': r.image_settings.color_depth,
            'use_motion_blur': r.use_motion_blur,
            'use_file_extension': r.use_file_extension,
            'film_transparent': r.film_transparent,
            'camera': scene.camera,
            'dof': [],
        }
        r.image_settings.file_format = self.image_format_eff
        if props.transparent_bg:
            # Keep the alpha channel so LichtFeld's alpha mask mode can isolate
            # the object; render a transparent background to fill it.
            r.image_settings.color_mode = 'RGBA'
            r.film_transparent = True
        else:
            r.image_settings.color_mode = 'RGB'
        if self.image_format_eff == 'PNG':
            r.image_settings.color_depth = '8'
        r.use_file_extension = True
        if self.maps_only:
            # Depth/normal/albedo passes are geometric — exact at 1 sample, so a
            # maps-only pass renders almost instantly regardless of beauty samples.
            if hasattr(scene, 'cycles'):
                snap['cycles_samples'] = scene.cycles.samples
                scene.cycles.samples = 1
            if hasattr(scene, 'eevee') and hasattr(scene.eevee, 'taa_render_samples'):
                snap['eevee_samples'] = scene.eevee.taa_render_samples
                scene.eevee.taa_render_samples = 1
        if props.disable_motion_blur:
            r.use_motion_blur = False
        if props.disable_dof:
            for cam in cams:
                snap['dof'].append((cam, cam.data.dof.use_dof))
                cam.data.dof.use_dof = False
        return snap

    def _restore_render_settings(self, scene, snap):
        r = scene.render
        r.filepath = snap['filepath']
        r.image_settings.file_format = snap['file_format']
        r.image_settings.color_mode = snap['color_mode']
        r.image_settings.color_depth = snap['color_depth']
        r.use_motion_blur = snap['use_motion_blur']
        r.use_file_extension = snap['use_file_extension']
        r.film_transparent = snap['film_transparent']
        if 'cycles_samples' in snap and hasattr(scene, 'cycles'):
            scene.cycles.samples = snap['cycles_samples']
        if 'eevee_samples' in snap and hasattr(scene, 'eevee'):
            scene.eevee.taa_render_samples = snap['eevee_samples']
        if snap['camera'] is not None:
            scene.camera = snap['camera']
        for cam, val in snap.get('dof', []):
            try:
                cam.data.dof.use_dof = val
            except ReferenceError:
                pass

    # --- compositor passes (depth / normal / albedo) setup & teardown --------
    # Blender 5.0 reworked the compositor: scene.node_tree -> scene.compositing_node_group
    # (a node-group data block) and the Composite node -> a Group Output node. We detect
    # the API at runtime so this works on 5.0+ and still falls back on 4.x. All requested
    # passes are written, as 32-bit float EXR, by one File Output node.
    @staticmethod
    def _enable_pass(view_layer, state, attr_candidates):
        for attr in attr_candidates:
            if hasattr(view_layer, attr):
                state.setdefault('passes', []).append((attr, getattr(view_layer, attr)))
                setattr(view_layer, attr, True)
                return
        state.setdefault('passes', [])

    def _setup_passes(self, scene, view_layer):
        # (key, view-layer pass attrs, RenderLayers socket names, subfolder, item socket type)
        specs = []
        if self.want_depth:
            specs.append(('depth', ('use_pass_z', 'use_pass_depth'),
                          ('Depth', 'Z'), 'depths', 'FLOAT'))
        if self.want_normal:
            specs.append(('normal', ('use_pass_normal',),
                          ('Normal',), 'normals', 'VECTOR'))
        if self.want_albedo:
            specs.append(('albedo', ('use_pass_diffuse_color',),
                          ('Diffuse Color', 'DiffCol'), 'albedo', 'RGBA'))

        state = {'created': [], 'fouts': [], 'v5': hasattr(scene, 'compositing_node_group')}
        for _key, attrs, _socks, _sub, _t in specs:
            self._enable_pass(view_layer, state, attrs)

        if hasattr(scene.render, 'use_compositing'):
            state['orig_use_compositing'] = scene.render.use_compositing
            scene.render.use_compositing = True

        # Ensure a compositor tree (5.0 group data-block, or legacy scene.node_tree).
        if state['v5']:
            tree = scene.compositing_node_group
            state['orig_group'] = tree
            state['created_group'] = tree is None
            if tree is None:
                tree = bpy.data.node_groups.new("3DGS_Compositing", "CompositorNodeTree")
                scene.compositing_node_group = tree
        else:
            state['orig_use_nodes'] = scene.use_nodes
            scene.use_nodes = True
            tree = scene.node_tree
        state['tree'] = tree

        rl = next((n for n in tree.nodes if n.type == 'R_LAYERS'), None)
        if rl is None:
            rl = tree.nodes.new('CompositorNodeRLayers')
            state['created'].append(rl)

        # Keep the beauty image reaching the scene output if we built the tree
        # ourselves (never disturb a user's existing compositor graph).
        if state['v5']:
            if state['created_group']:
                gout = tree.nodes.new('NodeGroupOutput')
                state['created'].append(gout)
                tree.interface.new_socket(name='Image', in_out='OUTPUT',
                                          socket_type='NodeSocketColor')
                img = rl.outputs.get('Image')
                if img is not None and len(gout.inputs) > 0:
                    tree.links.new(img, gout.inputs[0])
        else:
            comp = next((n for n in tree.nodes if n.type == 'COMPOSITE'), None)
            if comp is None:
                comp = tree.nodes.new('CompositorNodeComposite')
                state['created'].append(comp)
                img = rl.outputs.get('Image')
                if img is not None:
                    tree.links.new(img, comp.inputs['Image'])

        # Blender 5.0: File Output node uses directory + file_name + file_output_items
        # (typed). One node per pass keeps the per-camera filename unambiguous.
        for key, _attrs, socks, sub, stype in specs:
            sock = None
            for s in socks:
                sock = rl.outputs.get(s)
                if sock is not None:
                    break
            if sock is None:
                self.report({'WARNING'}, "Render Layers has no '%s' pass; skipped" % key)
                continue

            fo = tree.nodes.new('CompositorNodeOutputFile')
            fo.label = "3DGS_" + key.upper()
            sub_dir = os.path.join(self.out, sub)
            os.makedirs(sub_dir, exist_ok=True)
            fo.directory = sub_dir
            try:
                fo.format.file_format = 'OPEN_EXR'
                fo.format.color_depth = '32'
            except Exception:
                pass
            try:
                fo.file_output_items.clear()
            except Exception:
                pass
            try:
                item = fo.file_output_items.new(stype, key)
                if stype == 'VECTOR':
                    item.vector_socket_dimensions = 3
            except Exception as exc:                       # noqa: BLE001
                self.report({'WARNING'}, "Could not add '%s' output (%s); skipped" % (key, exc))
                tree.nodes.remove(fo)
                continue
            if len(fo.inputs) == 0:
                self.report({'WARNING'}, "Could not create '%s' output socket; skipped" % key)
                tree.nodes.remove(fo)
                continue
            tree.links.new(sock, fo.inputs[len(fo.inputs) - 1])
            state['created'].append(fo)
            state['fouts'].append({'node': fo, 'subdir': sub, 'key': key})
        return state

    def _teardown_passes(self, scene, view_layer, state):
        tree = state.get('tree')
        try:
            if tree is not None:
                for node in state.get('created', []):
                    try:
                        tree.nodes.remove(node)
                    except Exception:
                        pass
        finally:
            if state.get('v5'):
                if state.get('created_group'):
                    try:
                        scene.compositing_node_group = state.get('orig_group')
                    except Exception:
                        pass
                    try:
                        bpy.data.node_groups.remove(tree)
                    except Exception:
                        pass
            elif 'orig_use_nodes' in state:
                scene.use_nodes = state['orig_use_nodes']
            if 'orig_use_compositing' in state:
                try:
                    scene.render.use_compositing = state['orig_use_compositing']
                except Exception:
                    pass
            for attr, val in state.get('passes', []):
                try:
                    setattr(view_layer, attr, val)
                except Exception:
                    pass

    # --- point cloud dispatch -----------------------------------------------
    def _build_point_cloud(self, context, props, cam_records):
        mode = props.pc_mode
        n = props.pc_num_points
        if mode == 'NONE':
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        if mode == 'DEPTH':
            for rec in cam_records:                       # resolve actual depth EXR files
                g = rec.get('depth_glob')
                if g:
                    matches = glob.glob(g)
                    rec['depth_path'] = matches[0] if matches else None
            xyz, rgb = pointcloud.points_from_depth(cam_records, n)
            if len(xyz) > 0:
                return xyz, rgb
            self.report({'WARNING'}, "Depth back-projection produced no points; "
                                     "falling back to surface sampling")
            mode = 'SURFACE'

        # Limit sampling to the region the cameras occupy, so stray/far geometry
        # elsewhere in the scene can't dominate (or blow the cloud to infinity).
        bounds = None
        if props.pc_limit_to_cameras and cam_records:
            centers = np.array([(-r['R'].T @ r['t']) for r in cam_records])
            bounds = sampling.region_bounds_from_centers(centers, props.pc_region_padding)

        objs = self._mesh_objects(context, props)
        if mode == 'SURFACE':
            xyz, rgb = pointcloud.sample_surface(context, n, objs, bounds=bounds)
            if len(xyz) > 0:
                return xyz, rgb
            self.report({'WARNING'}, "No mesh surfaces in the camera region; using random points")
            mode = 'RANDOM'
        if mode == 'RANDOM':
            return pointcloud.sample_random(context, n, objs, bounds=bounds)
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    def _mesh_objects(self, context, props):
        if props.pc_source == 'SELECTED':
            pool = context.selected_objects
        else:
            pool = context.scene.objects
        return [o for o in pool if o.type == 'MESH' and not o.hide_render]


class GS_OT_export_cameras_only(Operator):
    bl_idname = "gs_export.export_cameras_only"
    bl_label = "Export Cameras Only (no render)"
    bl_description = "Write the COLMAP model and point cloud without re-rendering images"
    bl_options = {'REGISTER'}

    def execute(self, context):
        return bpy.ops.gs_export.render_export(render=False)


class GS_OT_export_maps_only(Operator):
    bl_idname = "gs_export.export_maps_only"
    bl_label = "Export GT Maps Only (fast, keeps images)"
    bl_description = ("Render only the ground-truth depth/normal/albedo passes (1 sample) "
                      "for the camera collection — fast, and does NOT re-render or "
                      "overwrite the colour images")
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        p = context.scene.gs_export
        return p.export_depth or p.export_normal or p.export_albedo

    def execute(self, context):
        return bpy.ops.gs_export.render_export('INVOKE_DEFAULT', maps_only=True)


classes = (
    GS_OT_add_camera_array,
    GS_OT_prepare_walkthrough,
    GS_OT_bake_cameras_from_anim,
    GS_OT_cull_clipping_cameras,
    GS_OT_render_export,
    GS_OT_export_cameras_only,
    GS_OT_export_maps_only,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
