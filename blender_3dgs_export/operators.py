"""Operators: add a camera array, and render + export a COLMAP dataset."""

import os
import re
import math
import tempfile
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
            ('EVEN_DISTANCE', "Even Distance",
             "Evenly spaced by distance travelled (best for variable walking speed)"),
            ('FRAME_STEP', "Every N Frames", "Sample one camera every N frames"),
        ],
        default='EVEN_DISTANCE',
    )
    frame_step: IntProperty(name="Frame Step", default=5, min=1, max=1000)
    spacing: FloatProperty(name="Spacing", default=0.5, min=0.001, soft_max=10.0,
                           subtype='DISTANCE')
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
        layout.prop(self, "frame_step" if self.sampling_mode == 'FRAME_STEP' else "spacing")
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
                positions = []
                for f in frames:
                    scene.frame_set(f)
                    dg = context.evaluated_depsgraph_get()
                    loc = src.evaluated_get(dg).matrix_world.translation
                    positions.append((loc.x, loc.y, loc.z))
                idx = sampling.resample_indices_by_distance(positions, self.spacing)
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
# Render + export
# --------------------------------------------------------------------------- #
class GS_OT_render_export(Operator):
    bl_idname = "gs_export.render_export"
    bl_label = "Render & Export Dataset"
    bl_description = "Render every camera in the collection and write a COLMAP dataset"
    bl_options = {'REGISTER'}

    render: BoolProperty(default=True, options={'HIDDEN'})

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

        images_dir = os.path.join(out, "images")
        sparse_dir = os.path.join(out, "sparse", "0")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)

        do_render = self.render
        depth_mode = (props.pc_mode == 'DEPTH') and do_render
        ext = IMAGE_EXT[props.image_format]

        restore = {}
        depth_state = None
        tmp_depth_dir = None
        wm = context.window_manager
        try:
            if do_render:
                restore = self._apply_render_settings(scene, props, cams)
            if depth_mode:
                tmp_depth_dir = tempfile.mkdtemp(prefix="gs_depth_")
                depth_state = self._setup_depth_nodes(scene, context.view_layer, tmp_depth_dir)

            cameras_out = []          # COLMAP camera intrinsics (deduplicated)
            images_out = []           # COLMAP image poses
            frames_out = []           # transforms.json frames
            cam_records = []          # for depth back-projection
            intr_to_id = {}
            used_names = set()
            frame = scene.frame_current

            wm.progress_begin(0, len(cams))
            for i, cam in enumerate(cams):
                wm.progress_update(i)
                scene.camera = cam

                w, h, fx, fy, cx, cy = camera_utils.compute_intrinsics(cam.data, scene.render)
                key = (w, h, round(fx, 4), round(fy, 4), round(cx, 4), round(cy, 4))
                if key not in intr_to_id:
                    cam_id = len(cameras_out) + 1
                    intr_to_id[key] = cam_id
                    cameras_out.append({
                        'id': cam_id, 'model': 'PINHOLE', 'width': w, 'height': h,
                        'params': [fx, fy, cx, cy],
                    })
                cam_id = intr_to_id[key]

                stem = _sanitize(cam.name)
                base = stem
                n = 1
                while stem in used_names:
                    stem = f"{base}_{n}"
                    n += 1
                used_names.add(stem)
                image_name = stem + ext
                image_path = os.path.join(images_dir, image_name)

                if do_render:
                    scene.render.filepath = os.path.join(images_dir, stem)
                    if depth_mode:
                        depth_state['fout'].file_slots[0].path = f"{stem}_depth_"
                    print(f"[3DGS] Rendering {i + 1}/{len(cams)}: {image_name}")
                    bpy.ops.render.render(write_still=True)

                qvec, tvec, R_np, t_np = camera_utils.get_extrinsics(cam)
                images_out.append({
                    'id': i + 1, 'qvec': qvec, 'tvec': tvec,
                    'camera_id': cam_id, 'name': image_name,
                })

                # transforms.json: camera-to-world = Blender matrix_world (OpenGL frame).
                frames_out.append({
                    'file_path': "images/" + image_name,
                    'transform_matrix': [[float(v) for v in row] for row in cam.matrix_world],
                    'w': w, 'h': h, 'fl_x': fx, 'fl_y': fy, 'cx': cx, 'cy': cy,
                })

                if depth_mode:
                    depth_path = os.path.join(tmp_depth_dir, f"{stem}_depth_{frame:04d}.exr")
                    cam_records.append({
                        'depth_path': depth_path, 'image_path': image_path,
                        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'R': R_np, 't': t_np,
                    })
            wm.progress_end()

            # ---- Point cloud ----
            xyz, rgb = self._build_point_cloud(context, props, cam_records)
            print(f"[3DGS] Point cloud: {len(xyz)} points ({props.pc_mode})")

            # ---- Write COLMAP model ----
            colmap_io.write_model(sparse_dir, cameras_out, images_out, xyz, rgb,
                                  fmt=props.colmap_format)
            if props.write_ply:
                colmap_io.write_points_ply(os.path.join(sparse_dir, "points3D.ply"), xyz, rgb)

            # ---- Optional transforms.json (NeRF / Nerfstudio / instant-ngp) ----
            if props.write_transforms_json and frames_out:
                ply_ref = "sparse/0/points3D.ply" if props.write_ply else None
                transforms_io.write_transforms(os.path.join(out, "transforms.json"),
                                               frames_out, ply_path=ply_ref)
                print(f"[3DGS] Wrote transforms.json ({len(frames_out)} frames)")

        finally:
            if depth_state is not None:
                self._teardown_depth_nodes(scene, context.view_layer, depth_state)
            if tmp_depth_dir and os.path.isdir(tmp_depth_dir):
                shutil.rmtree(tmp_depth_dir, ignore_errors=True)
            if restore:
                self._restore_render_settings(scene, restore)

        msg = (f"Exported {len(images_out)} views, {len(cameras_out)} intrinsic(s), "
               f"{len(xyz)} points -> {out}")
        self.report({'INFO'}, msg)
        print(f"[3DGS] {msg}")
        return {'FINISHED'}

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
            'camera': scene.camera,
            'dof': [],
        }
        r.image_settings.file_format = props.image_format
        r.image_settings.color_mode = 'RGB'
        if props.image_format == 'PNG':
            r.image_settings.color_depth = '8'
        r.use_file_extension = True
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
        if snap['camera'] is not None:
            scene.camera = snap['camera']
        for cam, val in snap.get('dof', []):
            try:
                cam.data.dof.use_dof = val
            except ReferenceError:
                pass

    # --- depth compositor setup / teardown ----------------------------------
    # Blender 5.0 reworked the compositor: scene.node_tree -> scene.compositing_node_group
    # (a node-group data block) and the Composite node -> a Group Output node. We detect
    # the API at runtime so the depth mode works on 5.0+ and still falls back on 4.x.
    @staticmethod
    def _enable_depth_pass(view_layer, state):
        # RNA property for the Z/Depth pass (name kept defensive across versions).
        for attr in ('use_pass_z', 'use_pass_depth'):
            if hasattr(view_layer, attr):
                state['pass_attr'] = attr
                state['orig_pass'] = getattr(view_layer, attr)
                setattr(view_layer, attr, True)
                return
        state['pass_attr'] = None

    def _setup_depth_nodes(self, scene, view_layer, tmp_dir):
        state = {'created': [], 'v5': hasattr(scene, 'compositing_node_group')}
        self._enable_depth_pass(view_layer, state)

        # Make sure the compositor actually runs at render time.
        if hasattr(scene.render, 'use_compositing'):
            state['orig_use_compositing'] = scene.render.use_compositing
            scene.render.use_compositing = True

        if state['v5']:
            # Blender 5.0+: compositor is an independent node-group data block.
            tree = scene.compositing_node_group
            state['orig_group'] = tree
            state['created_group'] = tree is None
            if tree is None:
                tree = bpy.data.node_groups.new("3DGS_Compositing", "CompositorNodeTree")
                scene.compositing_node_group = tree
        else:
            # Blender 3.3 - 4.x: compositor lives directly on the scene.
            state['orig_use_nodes'] = scene.use_nodes
            scene.use_nodes = True
            tree = scene.node_tree
        state['tree'] = tree

        # Render Layers node — the source of the Depth pass.
        rl = next((n for n in tree.nodes if n.type == 'R_LAYERS'), None)
        if rl is None:
            rl = tree.nodes.new('CompositorNodeRLayers')
            state['created'].append(rl)

        # If we created the compositor tree ourselves, wire the beauty image to the
        # scene output so the rendered PNG isn't black. Never touch a user's own graph.
        if state['v5']:
            if state['created_group']:
                out = tree.nodes.new('NodeGroupOutput')
                state['created'].append(out)
                tree.interface.new_socket(name='Image', in_out='OUTPUT',
                                          socket_type='NodeSocketColor')
                img_sock = rl.outputs.get('Image')
                if img_sock is not None and len(out.inputs) > 0:
                    tree.links.new(img_sock, out.inputs[0])
        else:
            comp = next((n for n in tree.nodes if n.type == 'COMPOSITE'), None)
            if comp is None:
                comp = tree.nodes.new('CompositorNodeComposite')
                state['created'].append(comp)
                img_sock = rl.outputs.get('Image')
                if img_sock is not None:
                    tree.links.new(img_sock, comp.inputs['Image'])

        # File Output node that writes the depth pass to OpenEXR.
        fout = tree.nodes.new('CompositorNodeOutputFile')
        fout.label = "3DGS_DEPTH"
        fout.base_path = tmp_dir
        fout.format.file_format = 'OPEN_EXR'
        fout.format.color_mode = 'RGB'
        fout.format.color_depth = '32'
        depth_sock = rl.outputs.get('Depth') or rl.outputs.get('Z')
        if depth_sock is None:
            raise RuntimeError("Render Layers node has no Depth output")
        tree.links.new(depth_sock, fout.inputs[0])
        state['created'].append(fout)
        state['fout'] = fout
        return state

    def _teardown_depth_nodes(self, scene, view_layer, state):
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
            if state.get('pass_attr'):
                try:
                    setattr(view_layer, state['pass_attr'], state['orig_pass'])
                except Exception:
                    pass

    # --- point cloud dispatch -----------------------------------------------
    def _build_point_cloud(self, context, props, cam_records):
        mode = props.pc_mode
        n = props.pc_num_points
        if mode == 'NONE':
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        if mode == 'DEPTH':
            xyz, rgb = pointcloud.points_from_depth(cam_records, n)
            if len(xyz) > 0:
                return xyz, rgb
            self.report({'WARNING'}, "Depth back-projection produced no points; "
                                     "falling back to surface sampling")
            mode = 'SURFACE'

        objs = self._mesh_objects(context, props)
        if mode == 'SURFACE':
            xyz, rgb = pointcloud.sample_surface(context, n, objs)
            if len(xyz) > 0:
                return xyz, rgb
            self.report({'WARNING'}, "No mesh surfaces to sample; using random points")
            mode = 'RANDOM'
        if mode == 'RANDOM':
            return pointcloud.sample_random(context, n, objs)
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


classes = (
    GS_OT_add_camera_array,
    GS_OT_prepare_walkthrough,
    GS_OT_bake_cameras_from_anim,
    GS_OT_render_export,
    GS_OT_export_cameras_only,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
