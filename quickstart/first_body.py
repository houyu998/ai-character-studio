"""Quick start: empty scene -> first MPFB body from a size preset -> saved .blend + two grey renders.

Run (Blender 4.4 with MPFB 2 installed and enabled):
    blender -b --factory-startup --addons bl_ext.user_default.mpfb --python quickstart/first_body.py -- --size M --out my_character

Writes <out>/versions/s1_draft_v001.blend and <out>/renders/v001_front.png, v001_side.png. Never overwrites:
it stops if any of these already exist.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import s1_core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="M", help="XS S M L XL 2XL ... (presets/body_sizes.json)")
ap.add_argument("--out", required=True, help="new character folder")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

out = os.path.abspath(args.out)  # Blender resolves relative paths oddly; always use absolute ones
blend = os.path.join(out, "versions", "s1_draft_v001.blend")
pngs = [os.path.join(out, "renders", f"v001_{v}.png") for v in ("front", "side")]
for p in [blend] + pngs:
    assert not os.path.exists(p), "exists, not overwriting: " + p
os.makedirs(os.path.dirname(blend), exist_ok=True)
os.makedirs(os.path.dirname(pngs[0]), exist_ok=True)

presets = json.load(open(os.path.join(REPO, "presets", "body_sizes.json"), encoding="utf-8"))
if args.size not in presets["sizes"]:
    sys.exit("unknown size %r, choose from %s" % (args.size, list(presets["sizes"])))

# 1. empty scene
s1_core.enable_mpfb()
import s1_custom  # noqa: E402  project controls on top of MPFB's
import s1_fit  # noqa: E402
s1_custom.register()

# 2. MPFB base human (female base macros; clears the scene first), then the size preset
body = s1_fit.build_base()
body.name = "CS_Body"
s1_fit.Fitter(body, s1_core.compute_zones(body, s1_core.raw_coords(body))).preload()
state = s1_core.apply_state(body, presets["sizes"][args.size]["state"])
print("K body", body.name, "verts", len(body.data.vertices), "size", args.size)

# 3. grey studio renders
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {e.identifier for e in
    bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items} else "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 800, 1200
w = bpy.data.worlds.new("Grey"); w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (.35, .36, .38, 1); sc.world = w
for name, energy, rot in [("Key", 3.0, (55, 0, -35)), ("Fill", 1.0, (70, 0, 40)), ("Rim", 2.0, (60, 0, 170))]:
    ld = bpy.data.lights.new(name, "SUN"); ld.energy = energy
    lo = bpy.data.objects.new(name, ld); sc.collection.objects.link(lo)
    lo.rotation_euler = [math.radians(x) for x in rot]
clay = bpy.data.materials.new("Clay")
body.data.materials.clear(); body.data.materials.append(clay)

dg = bpy.context.evaluated_depsgraph_get()
cs = [body.matrix_world @ v.co for v in body.evaluated_get(dg).data.vertices]
zmin, zmax = min(c.z for c in cs), max(c.z for c in cs)
centre = Vector((0, sum(c.y for c in cs) / len(cs), (zmin + zmax) / 2))
cd = bpy.data.cameras.new("Cam"); cd.type = "ORTHO"; cd.ortho_scale = (zmax - zmin) * 1.1
cam = bpy.data.objects.new("Cam", cd); sc.collection.objects.link(cam); sc.camera = cam
for p, yaw in zip(pngs, (0, 90)):
    d = Vector((math.sin(math.radians(yaw)), -math.cos(math.radians(yaw)), 0))
    cam.location = centre + d * 6
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = p
    bpy.ops.render.render(write_still=True)
    print("K wrote", p)

# 4. save the first version
bpy.ops.wm.save_as_mainfile(filepath=blend)
print("K saved", blend, "height_m", round(zmax - zmin, 3))
