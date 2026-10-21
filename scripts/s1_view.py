"""Stage-1 views: reference-vs-model silhouette overlay, three-view sheets, before/after sheets (run inside Blender)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
import s1_core as C

REF_PNG = os.path.join(C.REFDIR, "ref01_turnaround.png")
REF_MASK = os.path.join(C.REFDIR, "ref01_mask.png")        # binary silhouette of the reference (1536x1024)
PX_TOP, PX_FLOOR = 22, 969                                  # hair top / sole in the reference image
SKULL_TOP = 30                                              # skull top (hair adds ~8 px)
PANELS = {"front": (355, 145, 565), "side": (770, 560, 980), "back": (1200, 990, 1410)}   # (centre, x0, x1)
OUT_W, OUT_H = 420, 1024


def read_png(path):
    im = bpy.data.images.load(path, check_existing=False)
    w, h = im.size
    a = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, im.channels)
    bpy.data.images.remove(im)
    return a[::-1]                                          # top-down


def write_png(path, arr):
    arr = np.clip(arr, 0, 1).astype(np.float32)
    h, w = arr.shape[:2]
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), np.float32)], axis=2)
    im = bpy.data.images.new("tmp_out", w, h, alpha=True)
    im.pixels.foreach_set(arr[::-1].reshape(-1))
    im.filepath_raw = path
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)


def ref_mask():
    return read_png(REF_MASK)[:, :, 0] > 0.5


def ref_anchor(m):
    """Reference plumb / centre lines: ankle-band mid-points (fl 0.085..0.100) for each view."""
    out = {}
    Hh = PX_FLOOR - PX_TOP
    ys = [int(PX_FLOOR - f * Hh) for f in np.linspace(0.085, 0.100, 6)]
    for v, (pc, x0, x1) in PANELS.items():
        xs = []
        for y in ys:
            idx = np.where(m[y, x0:x1])[0]
            if len(idx):
                xs.append(0.5 * (idx.min() + idx.max()) + x0)
        out[v] = float(np.mean(xs))
    return out


def _silhouette(obj, view, plumb, zmin, H, anchors):
    ppu = (PX_FLOOR - SKULL_TOP) / H                        # px per metre
    pc = PANELS[view][0]
    cz = zmin + (PX_FLOOR - OUT_H / 2.0) / ppu
    if view == "front":
        lat = (pc - anchors["front"]) / ppu
    elif view == "side":
        lat = -plumb - (anchors["side"] - pc) / ppu
    else:
        lat = (pc - anchors["back"]) / ppu
    tmp = os.path.join(bpy.app.tempdir, "sil_%s.png" % view)
    C.render_view(obj, tmp, view, zones=False, ortho=True, res=(OUT_W, OUT_H), hide_arms=True, silhouette=True,
                  center_x=lat, ortho_scale=OUT_H / ppu, center_z=cz)
    return read_png(tmp)[:, :, 0] > 0.5


def model_metrics(obj):
    co = C.raw_coords(obj)
    zone = C.load_zones(obj)
    prof = C.Profiler(obj, zone)
    return prof(co)


def overlay_sheet(obj, path, label=None):
    """Reference (red) vs model (cyan) silhouettes, three views side by side; overlap is light grey."""
    m = ref_mask()
    anchors = ref_anchor(m)
    mp = model_metrics(obj)
    tiles = []
    for view in ("front", "side", "back"):
        pc, x0, x1 = PANELS[view]
        sil = _silhouette(obj, view, mp["plumb"], mp["zmin"], mp["H"], anchors)
        # reference crop with the same 420 px window centred on the panel centre
        ref = np.zeros((OUT_H, OUT_W), bool)
        xa = pc - OUT_W // 2
        for i in range(OUT_W):
            xx = xa + i
            if 0 <= xx < m.shape[1]:
                ref[:, i] = m[:OUT_H, xx]
        img = np.zeros((OUT_H, OUT_W, 3), np.float32) + 0.08
        img[ref & ~sil] = (0.95, 0.30, 0.30)
        img[sil & ~ref] = (0.25, 0.75, 1.0)
        img[ref & sil] = (0.85, 0.85, 0.85)
        tiles.append(img)
    write_png(path, np.concatenate(tiles, axis=1))
    return mp


def three_view_sheet(obj, path, zones=False, res=(560, 900), views=("front", "side", "back", "q34")):
    tiles = []
    for v in views:
        tmp = os.path.join(bpy.app.tempdir, "tile_%s.png" % v)
        C.render_view(obj, tmp, v, zones=zones, res=res)
        tiles.append(read_png(tmp)[:, :, :3])
    write_png(path, np.concatenate(tiles, axis=1))


def compare_sheet(objs_states, path, views=("front", "side", "q34"), zones=False, res=(560, 900)):
    """Rows = states (before / after ...), columns = views. objs_states: list of (obj, state_dict, title)."""
    rows = []
    for obj, state, title in objs_states:
        C.apply_state(obj, state)
        tiles = []
        for v in views:
            tmp = os.path.join(bpy.app.tempdir, "cmp_%s.png" % v)
            C.render_view(obj, tmp, v, zones=zones, res=res)
            tiles.append(read_png(tmp)[:, :, :3])
        rows.append(np.concatenate(tiles, axis=1))
    write_png(path, np.concatenate(rows, axis=0))
