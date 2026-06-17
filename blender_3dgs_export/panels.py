"""Sidebar UI in the 3D Viewport (N-panel > 3DGS Export)."""

import bpy
from bpy.types import Panel

from .operators import _cameras_in_collection


class GS_PT_base:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3DGS Export"


class GS_PT_cameras(GS_PT_base, Panel):
    bl_idname = "GS_PT_cameras"
    bl_label = "Cameras"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gs_export

        layout.operator("gs_export.add_camera_array", icon='OUTLINER_OB_CAMERA')
        layout.prop(props, "camera_collection")

        cams = _cameras_in_collection(props.camera_collection)
        box = layout.box()
        if props.camera_collection is None:
            box.label(text="No collection assigned", icon='INFO')
        else:
            box.label(text=f"{len(cams)} camera(s) to export", icon='CAMERA_DATA')

        layout.operator("gs_export.cull_clipping_cameras", icon='TRASH')


class GS_PT_capture(GS_PT_base, Panel):
    bl_idname = "GS_PT_capture"
    bl_label = "Capture (Walkthrough)"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("gs_export.prepare_walkthrough", icon='CAMERA_DATA')
        col.operator("gs_export.bake_cameras_from_anim", icon='TRACKING')
        layout.label(text="Drive a path, then bake to cameras", icon='INFO')


class GS_PT_pointcloud(GS_PT_base, Panel):
    bl_idname = "GS_PT_pointcloud"
    bl_label = "Point Cloud"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gs_export
        layout.prop(props, "pc_mode")
        if props.pc_mode in {'SURFACE', 'RANDOM'}:
            layout.prop(props, "pc_source")
        if props.pc_mode != 'NONE':
            layout.prop(props, "pc_num_points")
        if props.pc_mode in {'SURFACE', 'RANDOM'}:
            layout.prop(props, "pc_limit_to_cameras")
            if props.pc_limit_to_cameras:
                layout.prop(props, "pc_region_padding")
        if props.pc_mode == 'DEPTH':
            layout.label(text="Experimental: renders a depth pass", icon='ERROR')
        layout.prop(props, "write_ply")


class GS_PT_output(GS_PT_base, Panel):
    bl_idname = "GS_PT_output"
    bl_label = "Output & Export"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.gs_export

        layout.prop(props, "output_dir")
        layout.prop(props, "up_axis")
        layout.prop(props, "colmap_format")
        row = layout.row()
        row.enabled = not props.transparent_bg   # transparency forces PNG
        row.prop(props, "image_format")
        layout.prop(props, "transparent_bg")
        layout.prop(props, "write_transforms_json")

        box = layout.box()
        box.label(text="Ground-Truth Maps (EXR, for supervised / 2DGS)", icon='RENDERLAYERS')
        col = box.column(align=True)
        col.prop(props, "export_depth")
        col.prop(props, "export_normal")
        col.prop(props, "export_albedo")

        col = layout.column(align=True)
        col.prop(props, "disable_dof")
        col.prop(props, "disable_motion_blur")

        r = scene.render
        scale = r.resolution_percentage / 100.0
        w = int(round(r.resolution_x * scale))
        h = int(round(r.resolution_y * scale))
        info = layout.box()
        info.label(text=f"Render size: {w} x {h} px", icon='IMAGE_DATA')
        info.label(text=f"Engine: {r.engine}")

        layout.separator()
        layout.operator("gs_export.render_export", icon='RENDER_STILL')
        layout.operator("gs_export.export_cameras_only", icon='EXPORT')


classes = (GS_PT_cameras, GS_PT_capture, GS_PT_pointcloud, GS_PT_output)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
