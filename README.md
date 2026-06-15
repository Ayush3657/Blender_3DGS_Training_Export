# 3DGS Training Export (COLMAP) — Blender Addon

Render multi-view datasets straight from Blender and export camera poses + an
initial point cloud in **COLMAP format**, ready to train 3D Gaussian Splatting in
**[LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)** (or the Inria
reference trainer, Nerfstudio, etc.).

Because your scene already has *exact* camera poses and geometry, you skip the
entire photogrammetry/SfM step. No COLMAP solve, no pose noise — the camera
intrinsics, extrinsics, and the seed point cloud are written directly from the
scene.

> Designed for the archviz workflow: House interiors rendered with a perfect
> pinhole camera (no lens distortion, no DoF, fixed exposure).

---

## What it does

1. **Add Camera Array** — drop a grid / line / arc / dome of cameras into a
   dedicated collection, then nudge them into place by hand.
2. **Render & Export** — renders every camera in that collection with your
   current render settings and writes a COLMAP dataset:

   ```
   <output>/
     images/                 # one rendered image per camera
       GSCam_000.png ...
     sparse/0/
       cameras.txt  cameras.bin     # PINHOLE intrinsics
       images.txt   images.bin      # world-to-camera poses (OpenCV convention)
       points3D.txt points3D.bin    # initial colored point cloud
       points3D.ply                 # (optional)
     transforms.json                # (optional) NeRF / Nerfstudio / instant-ngp
   ```

3. Point this folder at a 3DGS training tool like LichtFeld Studio → train. (Or feed `transforms.json` to
   Nerfstudio / instant-ngp.)

---

## Install (Blender 5.0+)

Ships as a Blender **extension** (`blender_manifest.toml` included). Install it as
a zip.

**Windows (PowerShell)** — from this folder:

```powershell
Compress-Archive -Path .\blender_3dgs_export -DestinationPath .\blender_3dgs_export.zip -Force
```

This produces a zip with the `blender_3dgs_export/` folder inside (matching the
manifest `id`) — a valid extension package layout.

Then in Blender 5.0+:

1. **Edit ▸ Preferences ▸ Add-ons ▸ ▾ (top-right) ▸ Install from Disk…**
2. Pick `blender_3dgs_export.zip`.
3. It appears under **Add-ons** as **“3DGS Training Export (COLMAP)”** — enable it.
4. Open the **N-panel** in the 3D Viewport → **“3DGS Export”** tab.

No external dependencies — uses only `bpy`, `mathutils`, and the `numpy` bundled
inside Blender.

### Blender version notes

- **Target: Blender 5.0 and above.** The manifest sets `blender_version_min = 5.0.0`.
- The addon is written to also run on **4.2 – 4.5** (it detects the old vs. new
  compositor API at runtime). To install on 4.2–4.5, lower `blender_version_min`
  in `blender_3dgs_export/blender_manifest.toml` and re-zip.
- 5.0-specific handling baked in: the reworked compositor
  (`scene.compositing_node_group` + *Group Output* node, replacing
  `scene.node_tree` + *Composite*) is used for the optional depth point-cloud mode;
  the renamed **Depth** pass socket (was *Z*) and the `BLENDER_EEVEE` engine id (was
  `BLENDER_EEVEE_NEXT`) are handled too.

---

## Usage

### 1. Place cameras
- **Cameras** panel → **Add Camera Array**. Choose a pattern (the dialog centers
  it on the 3D cursor by default), set count/spacing/radius, and optionally aim
  them at the center.
- Cameras land in the **`3DGS_Cameras`** collection. Move/rotate/duplicate them
  freely, or add your own — anything in that collection gets exported.
- The panel shows the live camera count.

### 2. Choose the point cloud
**Point Cloud** panel:
- **Sample Mesh Surfaces** *(default)* — area-weighted points across mesh
  surfaces, colored from each material's base color. Fast, robust, accurate
  positions. Best for synthetic interiors.
- **Back-project Depth** *(experimental)* — renders a depth pass per camera and
  unprojects every pixel into a colored point. Densest, view-accurate. Slower;
  temporarily adds a depth output to the compositor and removes it afterwards.
- **Random in Bounds** — cheap fallback.
- **None** — empty cloud (LichtFeld expects a cloud, so avoid unless testing).

`Point Count` is the target total.

### 3. Output & export
**Output & Export** panel:
- **Output Folder** — dataset root (save your .blend first, or use an absolute
  path).
- **COLMAP Format** — `Both` writes `.bin` + `.txt` (recommended; loaders prefer
  `.bin`, the `.txt` is there for eyeballing).
- **Also Write transforms.json** — on by default; writes a NeRF-style
  `transforms.json` at the dataset root next to the COLMAP model (see below).
- **Disable Depth of Field / Motion Blur** — on by default; both break multi-view
  consistency. They're toggled only for the export and restored afterward.
- Hit **Render & Export Dataset**.

Already rendered? Use **Export Cameras Only** to (re)write the COLMAP model and
point cloud without re-rendering.

> Render settings are shared across all cameras (engine, samples, resolution,
> exposure, color management) — exactly what 3DGS wants. The addon only swaps the
> active camera between frames; it never changes your look.

---

## Feeding LichtFeld Studio

Point LichtFeld at `<output>` (the folder containing `images/` and `sparse/`).
It reads the COLMAP model, converts the OpenCV poses to its internal convention,
and initializes Gaussians from `points3D`.

If LichtFeld ever rejects the model, switch **COLMAP Format** to `Binary` (or
`Text`) and re-export — `Both` is the safe default precisely because different
loaders prefer different variants.

---

## transforms.json (Nerfstudio / instant-ngp / NeRF)

With **Also Write transforms.json** enabled, a `transforms.json` is written at the
dataset root, so the same folder works in NeRF-family tools too:

```bash
# Nerfstudio (3DGS via splatfacto)
ns-train splatfacto --data <output>
# instant-ngp: open <output> directly
```

Details:
- `transform_matrix` per frame is the **camera-to-world** matrix in the
  **OpenGL/Blender** convention (+X right, +Y up, −Z forward) — i.e. the camera's
  `matrix_world` dumped directly. This is the same convention the original NeRF
  Blender script used, so no axis conversion is applied (contrast with the COLMAP
  path, which stores world-to-camera in OpenCV axes).
- `camera_model` is `OPENCV` with zero distortion (perfect pinhole). Both modern
  intrinsics (`fl_x/fl_y/cx/cy/w/h`) and legacy `camera_angle_x/y` are written.
- Cameras with differing focal lengths get per-frame intrinsic overrides.
- If **Also Write points3D.ply** is on, `transforms.json` references it via
  `ply_file_path` so Nerfstudio's splatfacto can initialize from your cloud.

The COLMAP model and `transforms.json` describe the *same* cameras — verified in
`tests/` that both round-trip to the identical camera centers.

---

## How the conversion works (correctness notes)

These are the things that usually break direct exports; all are unit-tested in
`tests/`:

- **Intrinsics**: `PINHOLE` model. `fx = f_mm · width / sensor_width` (HORIZONTAL
  fit), with full handling of `sensor_fit` AUTO/HORIZONTAL/VERTICAL, pixel aspect,
  resolution %, and lens shift → principal point.
- **Extrinsics**: poses are **world-to-camera** in **OpenCV convention**
  (`+X` right, `+Y` down, `+Z` forward). Blender's camera (`-Z` forward, `+Y` up)
  is converted with `R_bcam→cv = diag(1, −1, −1)` applied to the inverse of
  `matrix_world`, then stored as quaternion + translation. Verified:
  `−Rᵀt` equals the Blender camera world position.
- **points3D stay in raw Blender world coordinates** — the OpenCV flip lives
  entirely in the baked rotation, so points and poses are consistent. (This is the
  single most common mistake in hand-rolled exporters.)
- **Binary format** matches COLMAP's `read_write_model.py` byte-for-byte
  (little-endian, packed records).

### Tests

The pure-math/IO modules are covered without needing Blender:

```bash
pip install numpy
python tests/test_colmap_io.py     # bin/txt round-trip vs a reference COLMAP reader
python tests/test_camera_math.py   # intrinsics + extrinsics invariants
python tests/test_geometry.py      # depth unprojection + surface sampling
python tests/test_transforms.py    # transforms.json + COLMAP/NeRF pose consistency
```

---

## Tips for clean 3DGS of interiors

- Keep **exposure / color management fixed** across all views (don't auto-expose).
- Use **enough overlap** between adjacent cameras — Gaussian splatting needs each
  surface seen from several angles. Dome/arc rings around a focal area help.
- Avoid pure mirrors / perfectly clear glass — view-dependent specular violates
  the multi-view-consistency assumption (some of it is fine; a hall of mirrors is
  not).
- For a whole room, scatter cameras throughout (the GRID pattern) and aim them
  outward/around rather than all parallel.
- Render at the resolution you'll train at; very high res = slower training with
  little quality gain for flat archviz.
