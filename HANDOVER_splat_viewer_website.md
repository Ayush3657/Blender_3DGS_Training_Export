# Handover: Real-Estate Gaussian-Splat Viewer Website (Demo)

> Self-contained brief for a **new project/conversation**. Goal: build a demo
> real-estate website where visitors can view a photoreal 3D Gaussian Splat of
> an interior with intuitive controls — **desktop and phone** — hosted cheaply
> as a static site. Copy this file into the new project folder and start there.

---

## 1. Who & why

**Ayush (GitHub `Ayush3657`, ASBL)** — archviz. He produces photoreal interior
renders in Blender/Cycles and is building **gamelike, streamable archviz**:
train 3D Gaussian Splats of flats, let prospective buyers explore them in the
browser. He previously did this in **Unreal Engine 5 with pixel streaming and
abandoned it** — hosting was expensive and capped concurrent users. The entire
point of the splat approach is: *photoreal, runs client-side, static hosting,
effectively unlimited users at near-zero cost.* Keep that constraint sacred.

Working style: he likes honest trade-off discussions, incremental verification
on real data, and being told when something is untested. Windows 11 dev
machine, RTX 5090, Node/npm availability unverified — check.

## 2. What already exists (upstream pipeline — a separate project)

- **Blender addon** ([github.com/Ayush3657/Blender_3DGS_Training_Export](https://github.com/Ayush3657/Blender_3DGS_Training_Export))
  exports perfect COLMAP datasets from Blender interiors (exact poses, GT
  depth/normal maps, depth-fusion init script). Do **not** rebuild any of this.
- Training happens in **LichtFeld Studio** (compiled from source,
  [github.com/MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio)),
  strategies IGS+/MCMC. A living-room scene (~1982 views) is trained/training
  now with a dense lit init; earlier runs reached ~3.5M splats, loss ~0.0197.
- Output: standard **3DGS `.ply`** export from LichtFeld (confirm in-app which
  extra formats his build exports — possibly compressed `.spz`/`.sog`).
- **The user supplies the trained splat file.** First session: ask him for the
  exported `.ply` (or export instructions) and its size.

## 3. Tech direction (already decided — don't relitigate)

- **Renderer: Spark** ([sparkjs.dev](https://sparkjs.dev),
  `github.com/sparkjsdev/spark`) — Three.js-based gaussian splat renderer. The
  user explicitly chose "Spark 2.0" for streaming/perf. Verify current API +
  supported formats from its docs at build time. Only propose an alternative
  (PlayCanvas/SuperSplat, gsplat.js, Babylon) if Spark has a genuine blocker;
  say so explicitly if it does.
- **Stack suggestion:** Vite + Three.js + Spark, plain TS/JS, static deploy
  (Cloudflare Pages / Netlify / Vercel — pick with the user). No backend.

## 4. Hard technical facts the next session should NOT rediscover

1. **Splat up-axis is −Y.** The datasets were exported Blender Z-up → **−Y up**
   (that's what displays upright in LichtFeld; COLMAP/OpenCV-style gravity).
   Three.js is **+Y up**, so the loaded splat will appear **upside down**
   without a 180° rotation about X (or Z). Fix once at load
   (e.g. `mesh.rotation.x = Math.PI`), verify visually, and keep floor-level
   (`y`) offset configurable.
2. **Raw PLY is huge.** Standard 3DGS PLY ≈ ~248 bytes/splat (62 floats incl.
   45 SH coefficients). 3.5M splats ≈ **~870 MB** — unusable on web/mobile.
   Compression is a *core workstream*, not a nice-to-have:
   - Convert to a compressed web format (`.spz` / SOG / `.ksplat` — whichever
     Spark supports best at build time; verify).
   - Strip/reduce spherical harmonics (SH0 or degree-1) — big win, small
     visual cost indoors.
   - Ask him to train web scenes with a tighter budget (1–2M gaussians via
     IGS+ — it's specifically efficient per-gaussian; he knows this).
   - Target: **≤ 30–80 MB** per scene over the wire; less for mobile.
3. **Scene content:** interiors are room-scale (metres), but the splat may
   contain **through-window exterior content** out to ~30–50 m (visible city
   through balcony/windows). Camera bounds and near/far planes should assume
   room-scale with distant background.
4. There may be a stray view named `Camera` in older datasets — irrelevant to
   the viewer, just don't be confused by it.

## 5. Demo requirements

**Viewer (the product):**
- Desktop: intuitive orbit/pan/zoom by default; optional first-person **walk
  mode** (WASD + mouse-look) for the gamelike feel. Clamp camera to sensible
  bounds (stay roughly inside the flat; no flying through walls — simple AABB
  or radius clamp is fine for the demo, no physics).
- **Mobile:** one-finger look/orbit, two-finger pinch-zoom/pan; consider
  optional gyro look. Must run acceptably on a mid-range phone: clamp
  devicePixelRatio, cap resolution, use the compressed/SH-stripped asset,
  consider a lower-splat-count mobile variant.
- Loading UX: progress indicator, ideally progressive/streamed loading; the
  page must feel alive before the splat finishes downloading.

**Website wrapper (thin, demo-grade):**
- A clean real-estate landing: property hero, key stats (beds/area/etc. —
  placeholder), a prominent "Explore in 3D" viewer (embedded + fullscreen),
  optionally a scene switcher for multiple rooms/flats later. Modern minimal
  design; mobile-first. No auth/CMS/payment — it's a demo.
- Ask about branding (ASBL?) and design taste before styling.

**Explicitly out of scope now (but don't architect against it):**
- Interactive objects: doors opening (per-object splats parented to
  transforms), **day/night toggle** as two overlapping splats crossfaded with
  an outward-gradient opacity animation, furniture toggles. These are the
  user's Phase-3 vision — structure the viewer code so per-object splats and
  state crossfades can be added later (multiple splat instances in one scene,
  per-instance transforms/opacity already supported by Spark — verify).
- The exterior drone-city splat composited behind windows.

## 6. Suggested first-session plan

1. Ask for: the trained `.ply` (+ size), LichtFeld export options he sees,
   branding/design preferences, priority of walk-mode vs orbit, target phone.
2. Get the asset web-ready: convert/compress, measure before/after, visual
   sanity check (and the −Y-up flip).
3. Scaffold Vite + Spark viewer; desktop controls; verify orientation/scale.
4. Mobile controls + perf pass on a real phone.
5. Wrap in the demo real-estate page; deploy to static hosting; hand over URL.

Verify each step on the real asset before moving on — that's how this
collaboration has worked so far and it has caught every major bug.

## 7. Reference pointers

| Thing | Where |
|---|---|
| Pipeline repo (addon, scripts, this file) | `D:\Ai_Tools\Blender_3DGS_Training_Export` / [GitHub](https://github.com/Ayush3657/Blender_3DGS_Training_Export) |
| Example trained dataset (COLMAP + images + depths) | `D:\Legacy\Interior\Data_Livin` |
| LichtFeld Studio | [github.com/MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) |
| Spark renderer | [sparkjs.dev](https://sparkjs.dev) · [github.com/sparkjsdev/spark](https://github.com/sparkjsdev/spark) |
| DN-Splatter clone (unused, parked) | `D:\Softwares\DN_Splatter` |
