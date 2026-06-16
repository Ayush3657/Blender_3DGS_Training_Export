"""Scene-level settings for the 3DGS export addon (stored at scene.gs_export)."""

import bpy
from bpy.props import (
    StringProperty, EnumProperty, IntProperty, FloatProperty, BoolProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup


class GS_ExportSettings(PropertyGroup):
    output_dir: StringProperty(
        name="Output Folder",
        description="Dataset root. images/ and sparse/0/ are created inside it",
        subtype='DIR_PATH',
        default="//gs_dataset/",
    )

    camera_collection: PointerProperty(
        name="Camera Collection",
        description="Every camera object in this collection is rendered and exported. "
                    "Use 'Add Camera Array' to populate it, or assign your own",
        type=bpy.types.Collection,
    )

    colmap_format: EnumProperty(
        name="COLMAP Format",
        description="Which sparse model files to write",
        items=[
            ('BOTH', "Both (.bin + .txt)", "Write both; loaders prefer .bin, .txt is human-readable"),
            ('BIN', "Binary (.bin)", "Canonical COLMAP binary files only"),
            ('TXT', "Text (.txt)", "Human-readable text files only"),
        ],
        default='BOTH',
    )

    up_axis: EnumProperty(
        name="Up Axis",
        description="World up axis of the exported dataset. Blender is Z-up; most 3DGS "
                    "viewers (LichtFeld Studio, OpenGL) are Y-up, so Z-up scenes appear "
                    "tipped on their side. Points and camera poses are rotated together",
        items=[
            ('Y_UP', "Y up (LichtFeld / most viewers)",
             "Convert Blender Z-up to Y-up so the scene appears upright"),
            ('Z_UP', "Z up (Blender, no change)",
             "Keep Blender's native Z-up orientation"),
        ],
        default='Y_UP',
    )

    image_format: EnumProperty(
        name="Image Format",
        items=[
            ('PNG', "PNG", "Lossless 8-bit PNG (recommended)"),
            ('JPEG', "JPEG", "Smaller files, lossy"),
        ],
        default='PNG',
    )

    pc_mode: EnumProperty(
        name="Point Cloud",
        description="How to generate the initial points3D cloud",
        items=[
            ('SURFACE', "Sample Mesh Surfaces", "Area-weighted samples on mesh surfaces (recommended)"),
            ('DEPTH', "Back-project Depth", "Unproject rendered depth maps (experimental, slower)"),
            ('RANDOM', "Random in Bounds", "Uniform random points in the scene bounding box"),
            ('NONE', "None", "Write an empty point cloud (not recommended)"),
        ],
        default='SURFACE',
    )

    pc_source: EnumProperty(
        name="Sample From",
        description="Which mesh objects to sample for the point cloud",
        items=[
            ('VISIBLE', "All Render-Visible", "All mesh objects not disabled for rendering"),
            ('SELECTED', "Selected Only", "Only selected mesh objects"),
        ],
        default='VISIBLE',
    )

    pc_num_points: IntProperty(
        name="Point Count",
        description="Target number of points in the initial cloud",
        default=200000, min=1000, max=20000000, soft_max=2000000,
    )

    pc_limit_to_cameras: BoolProperty(
        name="Limit to Camera Region",
        description="Only sample geometry near where the cameras are. Prevents stray "
                    "or far-away objects elsewhere in the scene from dominating the "
                    "cloud (or blowing it out to infinity)",
        default=True,
    )

    pc_region_padding: FloatProperty(
        name="Region Padding",
        description="Extra margin (metres) added around the camera bounding box when "
                    "limiting the sample region. Increase if walls/ceiling get cut off",
        default=2.0, min=0.0, soft_max=20.0, subtype='DISTANCE',
    )

    write_ply: BoolProperty(
        name="Also Write points3D.ply",
        description="Write a binary PLY copy of the point cloud (handy for previewing, "
                    "and used by Nerfstudio splatfacto for initialization)",
        default=False,
    )

    write_transforms_json: BoolProperty(
        name="Also Write transforms.json",
        description="Additionally export a NeRF / Nerfstudio / instant-ngp transforms.json "
                    "at the dataset root (alongside the COLMAP model)",
        default=True,
    )

    disable_dof: BoolProperty(
        name="Disable Depth of Field",
        description="Temporarily disable DoF on export cameras (blur breaks multi-view consistency)",
        default=True,
    )

    disable_motion_blur: BoolProperty(
        name="Disable Motion Blur",
        description="Temporarily disable motion blur during export",
        default=True,
    )


classes = (GS_ExportSettings,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gs_export = PointerProperty(type=GS_ExportSettings)


def unregister():
    del bpy.types.Scene.gs_export
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
