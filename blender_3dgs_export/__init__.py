"""3DGS Training Export — render multi-view datasets from Blender cameras and
export camera poses + an initial point cloud in COLMAP format, ready to train
3D Gaussian Splatting (Lichtfeld Studio, Inria gaussian-splatting, etc.).
"""

# On Blender 5.0+ this installs as an extension and blender_manifest.toml supplies the
# metadata (bl_info is ignored). bl_info is kept only for the legacy add-on path.
bl_info = {
    "name": "3DGS Training Export (COLMAP)",
    "author": "ASBL",
    "version": (1, 3, 1),
    "blender": (3, 3, 0),
    "location": "View3D > Sidebar (N) > 3DGS Export",
    "description": "Render cameras and export a COLMAP dataset for Gaussian Splatting training",
    "category": "Render",
}

# --- hot-reload support (re-running 'Install' or toggling the addon) ---------
if "bpy" in locals():
    import importlib
    for _m in ("camera_utils", "colmap_io", "pointcloud", "transforms_io",
               "sampling", "properties", "operators", "panels"):
        if _m in locals():
            importlib.reload(locals()[_m])

import bpy  # noqa: E402

from . import camera_utils  # noqa: F401,E402  (pure logic, no register)
from . import colmap_io      # noqa: F401,E402
from . import pointcloud     # noqa: F401,E402
from . import transforms_io  # noqa: F401,E402
from . import sampling       # noqa: F401,E402
from . import properties     # noqa: E402
from . import operators      # noqa: E402
from . import panels         # noqa: E402

_REGISTER_MODULES = (properties, operators, panels)


def register():
    for m in _REGISTER_MODULES:
        m.register()


def unregister():
    for m in reversed(_REGISTER_MODULES):
        m.unregister()


if __name__ == "__main__":
    register()
