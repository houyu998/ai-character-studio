"""v3 bust / glute PARTS (A3 零件): runtime overlay - s1_custom / a_points / s1_panel files are unchanged.

What a part is here (honest description):
  * the part region = the project's own anatomical zone (bust = ellipsoid cap around the nipple, glute = buttock band; built
    on the neutral body, stored as zone_id), split left / right; the midline (sternum, gluteal cleft) stays body.
  * the part OBJECT (CS_Bust_L/R pink, CS_Glute_L/R orange) is a separate, selectable, editable mesh made of exactly the body
    faces of that region (same vertex ids -> stable ids, metadata in custom properties).  The body gives way at DISPLAY time only:
    its faces of a visible part are deleted after the display smoothing (modifier "v3_trim", nothing deleted from the mesh -> fully
    recoverable, no double layer); border normals come from a hidden helper (see ensure_seam) so the border shows no shading step.
  * the part SHAPE lives in the body pipeline (a POST deformer "v3_parts", last in the chain): install transform (position /
    yaw / pitch / scale, in the part's own frame) blended over a transition band that is 0 on the interface ring, plus the part's
    shape deltas (user edits made on the part object, or a replacement source), stored in the part's local frame.
    Because the ring vertices never move, part and body share the same edge loop: the combined mesh is truly welded.
  * after every state change (weight, proportions ...) the part objects are refitted from the body (interface frame + band
    are recomputed from the current surface, the stored deltas ride along in the local frame).
  * replacement: sources with the SAME topology (assets/parts/*.json, local deltas per vertex id).  Importing an arbitrary
    foreign mesh is NOT supported (would need retopology / shrinkwrap) -> reported as PARTIAL.
"""
import os
import json
import math
import copy
import numpy as np
import bpy

import s1_core as C
import s1_custom as SC
import a_points as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "assets", "parts")
PROP = "v3_parts"                       # JSON on the body: {part id: {"e": {vid: [a,b,c]}, "source": str}}
MASK_VG = "v3_parts_hidden"
MASK_MOD = "v3_parts_mask"
FIT_ATTR = "v3_fit"
VID_ATTR = "v3_vid"
TRIM_ATTR = "v3_trim"                   # face attribute: True = this face is not drawn (deleted AFTER the display smoothing)
TRIM_MOD = "v3_trim"                    # Geometry Nodes modifier (last in the stack) that deletes those faces
TRIM_NG = "CS_v3_trim_faces"
SEAM_REF = "CS_v3_SeamRef"              # hidden display helper: SAME mesh data as the body (follows every change), smoothed like it, never drawn
SEAM_BAND_VG = "v3_seam_band"           # body vertices the helper keeps: the parts + SEAM_RINGS face rings (a whole-body helper costs ~2x more)
SEAM_NEAR_VG = "v3_seam_near"           # where the body takes the helper's normals: the parts + 1 ring (the borders and their neighbours)
SEAM_MOD = "v3_seam_normals"            # Data Transfer modifier after the trim: normals of the one continuous smoothed surface
SEAM_RINGS = 4
SEAM_MAX_DIST = 0.0005                  # m: only where the surfaces coincide (they do, to 0.0002 mm); anything else keeps its own normal
CAPTURE_TOL = 5e-5                      # m: part-object vertex moved more than this = a user edit

KINDS = {
    "bust": dict(zone="bust", band=0.035, label="胸", color=(0.93, 0.47, 0.66, 1.0), obj="CS_Bust_%s", mat="CS_Mat_Bust_Pink"),
    "glute": dict(zone="glute", band=0.045, label="臀", color=(0.97, 0.58, 0.22, 1.0), obj="CS_Glute_%s", mat="CS_Mat_Glute_Orange"),
}
PARTS = [k + "_" + s for k in KINDS for s in ("l", "r")]
BODY_MAT = ("CS_Mat_Body_Grey", (0.62, 0.62, 0.62, 1.0))
AX = [("x", "左右（+外）", "cm", -3.0, 3.0), ("y", "前后（+外凸）", "cm", -3.0, 3.0), ("z", "上下（+上）", "cm", -3.0, 3.0),
      ("yaw", "朝向 左右转（+朝外）", "°", -15.0, 15.0), ("pitch", "朝向 上下仰（+朝上）", "°", -15.0, 15.0),
      ("scale", "整体大小（范围缩放）", "", -0.2, 0.2)]


def cid(part, ax):
    k, s = part.split("_")
    return "v3_%s_%s_%s" % (k, s, ax)


def _controls():
    out = [dict(id="v3_parts_on", label="零件系统（胸/臀零件）", group="v3·零件安装", kind="custom", lo=0.0, hi=1.0, default=0.0,
                have="新增·v3", stage="A3", note="1 = 胸臀零件的安装/编辑/替换生效；0 = 与 v020 完全相同")]
    for k, kd in KINDS.items():
        out.append(dict(id="v3_%s_on" % k, label="%s零件形状生效" % kd["label"], group="v3·零件安装", kind="custom", lo=0.0, hi=1.0,
                        default=1.0, have="新增·v3", stage="A3",
                        note="0 = 卸下%s零件：该区域回到身体自身形状（安装量、编辑、替换都不作用，数据保留可恢复）" % kd["label"]))
    for p in PARTS:
        k, s = p.split("_")
        for ax, lab, unit, lo, hi in AX:
            out.append(dict(id=cid(p, ax), label="%s%s %s%s" % ("左" if s == "l" else "右", KINDS[k]["label"], lab, ("（%s）" % unit) if unit else ""),
                            group="v3·零件安装", kind="custom", lo=lo, hi=hi, default=0.0, have="新增·v3", stage="A3",
                            note="在零件自身坐标系里（原点=接口环中心，前=零件外凸方向，上=竖直）；接口环不动，过渡带内渐变"))
    return out


CONTROLS = _controls()

PD = {}                                  # live parts data (mirrors the body's PROP)
LAST = {}                                # part id -> (o, R, M) of the last deformer run (M = Rot * scale, local)
_GEO = {}                                # (obj name, nverts) -> static region data


# ------------------------------------------------------------------ static geometry
def _faces(obj):
    me = obj.data
    ls = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), dtype=np.int32); me.loops.foreach_get("vertex_index", lv)
    return ls, lt, lv


def geo(obj):
    key = (obj.name, len(obj.data.vertices))
    g = _GEO.get(key)
    if g is not None:
        return g
    zone = C.load_zones(obj)
    xr = A.rest_x(obj)
    ed = C._adjacency(obj)
    ea, eb = ed[:, 0], ed[:, 1]
    ls, lt, lv = _faces(obj)
    fid = np.repeat(np.arange(len(ls)), lt)
    g = {}
    for p in PARTS:
        k, s = p.split("_")
        R = (zone == C.ZID[KINDS[k]["zone"]]) & ((xr > 1e-4) if s == "l" else (xr < -1e-4))
        cross = R[ea] != R[eb]
        ring = np.zeros(len(R), bool)
        ring[ea[cross & R[ea]]] = True
        ring[eb[cross & R[eb]]] = True
        inner = R & ~ring
        fin = np.zeros(len(ls), bool)
        np.logical_or.at(fin, fid, inner[lv])
        faces = np.flatnonzero(fin)
        fv = [lv[ls[f]:ls[f] + lt[f]] for f in faces]
        vids = np.unique(np.concatenate(fv))
        er = R[ea] & R[eb]
        g[p] = dict(region=np.flatnonzero(R), ring=np.flatnonzero(ring), inner=np.flatnonzero(inner), faces=faces, fv=fv,
                    vids=vids, ea=ea[er], eb=eb[er], inner_mask=inner)
    _GEO[key] = g
    return g


def band_weight(co, g, band):
    """0 on the interface ring (and outside), smoothstep of the geodesic distance from the ring inside the region"""
    n = len(co)
    dist = np.full(n, np.inf)
    dist[g["ring"]] = 0.0
    ea, eb = g["ea"], g["eb"]
    L = np.linalg.norm(co[ea] - co[eb], axis=1)
    for _ in range(80):
        old = dist[g["region"]].copy()
        np.minimum.at(dist, eb, dist[ea] + L)
        np.minimum.at(dist, ea, dist[eb] + L)
        if np.allclose(old, dist[g["region"]]):
            break
    w = np.zeros(n)
    r = g["region"]
    w[r] = SC._ss(np.where(np.isfinite(dist[r]), dist[r], 0.0) / band)
    w[g["ring"]] = 0.0
    return w, dist


TRANS_FRAC = 1.0                        # transition band = the whole part (fraction of the largest ring distance) ...
TRANS_SMOOTH = 20                       # ... + Laplacian smoothing (exp v3_exp_band: 115 -> 10 new health warnings over 15 installs)


def trans_weight(co, g, dist):
    """install weight: 0 on the ring, smooth rise over the whole part (a narrow band made spikes / creases at 1.5 cm)"""
    r = g["region"]
    dr = np.where(np.isfinite(dist[r]), dist[r], 0.0)
    D = max(float(dr.max()), 1e-6)
    w = np.zeros(len(co))
    w[r] = SC._ss(dr / (TRANS_FRAC * D))
    ea, eb, ring = g["ea"], g["eb"], g["ring"]
    w[ring] = 0.0
    nb = np.zeros(len(co)); np.add.at(nb, ea, 1); np.add.at(nb, eb, 1)
    m = nb > 0
    for _ in range(TRANS_SMOOTH):
        sm = np.zeros(len(co)); np.add.at(sm, ea, w[eb]); np.add.at(sm, eb, w[ea])
        w[m] = 0.5 * w[m] + 0.5 * sm[m] / nb[m]
        w[ring] = 0.0
    w[r] /= max(float(w[r].max()), 1e-6)        # v3: the part centre (weight max) moves exactly by the install value (glute max was 0.37)
    return w


def frame(co, g, w, side):
    """interface frame: o = ring centre, f = towards the part core (its outward bulge), u = vertical, l = lateral outwards"""
    o = co[g["ring"]].mean(0)
    core = g["region"][w[g["region"]] > 0.999]
    if len(core) < 3:
        core = g["inner"]
    f = co[core].mean(0) - o
    f /= max(np.linalg.norm(f), 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    u = z - (z @ f) * f
    u /= max(np.linalg.norm(u), 1e-9)
    l = np.cross(f, u)
    if l[0] * (1.0 if side == "l" else -1.0) < 0:
        l = -l
    return o, np.column_stack([l, f, u])


def rot_local(yaw_deg, pitch_deg):
    """local (l, f, u): +yaw turns the apex outwards (f -> +l), +pitch turns it up (f -> +u)"""
    a, b = math.radians(yaw_deg), math.radians(pitch_deg)
    Y = np.array([[math.cos(a), math.sin(a), 0.0], [-math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]).T
    P = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(b), -math.sin(b)], [0.0, math.sin(b), math.cos(b)]])
    return P @ Y


# ------------------------------------------------------------------ deformer (POST, last)
def _st(obj):
    import v3_body as VB
    return VB._state(obj)


def d_v3_parts(co, ctx, amount):
    obj = bpy.data.objects[ctx["obj_name"]]
    st = _st(obj)
    G = geo(obj)
    d = np.zeros_like(co)
    for p in PARTS:
        k, s = p.split("_")
        on = float(st.get("v3_%s_on" % k, 1.0)) * float(amount)
        g = G[p]
        wb, dist = band_weight(co, g, KINDS[k]["band"])
        o, R = frame(co, g, wb, s)
        w = trans_weight(co, g, dist)
        v = {ax: float(st.get(cid(p, ax), 0.0)) for ax, *_ in AX}
        M = rot_local(v["yaw"], v["pitch"]) * (1.0 + v["scale"])
        t = np.array([v["x"], v["y"], v["z"]]) * 0.01
        LAST[p] = (o, R, M)
        r = g["region"]
        q = (co[r] - o) @ R                                  # local coordinates
        tgt = o + (q @ M.T + t) @ R.T
        d[r] += on * w[r][:, None] * (tgt - co[r])
        e = PD.get(p, {}).get("e")
        if e and on > 0:
            ids = np.array([int(i) for i in e], dtype=int)
            ok = g["inner_mask"][ids]
            ev = np.array(list(e.values()), dtype=np.float64)[ok]
            d[ids[ok]] += on * (ev @ M.T) @ R.T
    return d


# ------------------------------------------------------------------ parts data (edits / sources) + history
def load_data(obj):
    PD.clear()
    try:
        PD.update(json.loads(obj.get(PROP, "{}")))
    except ValueError:
        pass
    for p in PARTS:
        PD.setdefault(p, dict(e={}, source="原始（身体自身形状）"))
    return PD


def save_data(obj):
    obj[PROP] = json.dumps({p: dict(e={k: [round(x, 6) for x in v] for k, v in PD[p]["e"].items() if max(abs(x) for x in v) > 1e-7},
                                    source=PD[p]["source"]) for p in PARTS})


def snapshot(obj=None):
    return copy.deepcopy({p: PD[p] for p in PARTS if p in PD})


def restore(obj, snap):
    PD.clear()
    PD.update(copy.deepcopy(snap))
    for p in PARTS:
        PD.setdefault(p, dict(e={}, source="原始（身体自身形状）"))
    save_data(obj)


def has_parts(obj):
    return obj is not None and PROP in obj and any(bpy.data.objects.get(part_obj_name(p)) for p in PARTS)


def refit(obj):
    C.apply_state(obj, C.get_state(obj))


# ------------------------------------------------------------------ part objects
def part_obj_name(p):
    k, s = p.split("_")
    return KINDS[k]["obj"] % s.upper()


def _mat(name, col):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = col
    return m


def _skirt(obj, g):
    """faces of the part + one outer ring of body faces ("skirt") around it, with a per-face 'trim' flag on the skirt.
    The skirt is only there so the display smoothing of the part sees the same neighbourhood as the body's own smoothing (then both
    end on exactly the same smoothed border curve); it is deleted after the smoothing and never drawn."""
    ls, lt, lv = _faces(obj)
    fid = np.repeat(np.arange(len(ls)), lt)
    own = np.zeros(len(ls), bool); own[g["faces"]] = True
    vin = np.zeros(len(obj.data.vertices), bool); vin[g["vids"]] = True
    touch = np.zeros(len(ls), bool); np.logical_or.at(touch, fid, vin[lv])
    faces = np.concatenate([g["faces"], np.flatnonzero(touch & ~own)])
    fv = [lv[ls[f]:ls[f] + lt[f]] for f in faces]
    trim = np.concatenate([np.zeros(len(g["faces"]), bool), np.ones(len(faces) - len(g["faces"]), bool)])
    return faces, fv, trim, np.unique(np.concatenate(fv))


def _trim_group():
    ng = bpy.data.node_groups.get(TRIM_NG)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(TRIM_NG, 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    na = ng.nodes.new("GeometryNodeInputNamedAttribute")
    na.data_type = 'BOOLEAN'
    na.inputs["Name"].default_value = TRIM_ATTR
    dl = ng.nodes.new("GeometryNodeDeleteGeometry")
    dl.domain, dl.mode = 'FACE', 'ALL'
    ng.links.new(gi.outputs[0], dl.inputs["Geometry"])
    ng.links.new(na.outputs["Attribute"], dl.inputs["Selection"])
    ng.links.new(dl.outputs["Geometry"], go.inputs[0])
    return ng


def _tail(o):
    """stack order: ... display smoothing, trim, border normals (last)"""
    for name in (TRIM_MOD, SEAM_MOD):
        m = o.modifiers.get(name)
        if m is not None:
            i = list(o.modifiers).index(m)
            if i != len(o.modifiers) - 1:
                o.modifiers.move(i, len(o.modifiers) - 1)


def _trim_mod(o):
    """'delete the faces flagged v3_trim' after the display smoothing (only the border-normal modifier comes later)"""
    m = o.modifiers.get(TRIM_MOD)
    if m is None:
        m = o.modifiers.new(TRIM_MOD, 'NODES')
    if m.node_group is None:
        m.node_group = _trim_group()
    m.show_on_cage = False
    _tail(o)
    return m


def _grow(obj, vmask, n):
    """vertex mask grown by n rings of faces"""
    ls, lt, lv = _faces(obj)
    fid = np.repeat(np.arange(len(ls)), lt)
    for _ in range(n):
        f = np.zeros(len(ls), bool)
        np.logical_or.at(f, fid, vmask[lv])
        vmask = vmask.copy()
        vmask[lv[f[fid]]] = True
    return vmask


def _set_group(obj, name, vmask):
    vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    vg.remove(list(range(len(obj.data.vertices))))
    vg.add([int(i) for i in np.flatnonzero(vmask)], 1.0, 'REPLACE')


def _drive_smooth(mod, obj):
    """a helper's / part's display smoothing follows the body's (「显示平滑」 toggle, levels): a mismatch would reopen the border"""
    for prop in ("show_viewport", "show_render", "levels", "render_levels"):
        mod.driver_remove(prop)
        d = mod.driver_add(prop).driver
        d.type = 'AVERAGE'                                   # one variable, no Python expression (works without script auto-run)
        v = d.variables.new()
        v.name, v.type = "v", 'SINGLE_PROP'
        v.targets[0].id_type = 'OBJECT'
        v.targets[0].id = obj
        v.targets[0].data_path = 'modifiers["%s"].%s' % (C.DISPLAY_SMOOTH, prop)


def _seam_ref(obj):
    G = geo(obj)
    fp = np.zeros(len(obj.data.vertices), bool)
    for p in PARTS:
        fp[G[p]["vids"]] = True
    _set_group(obj, SEAM_BAND_VG, _grow(obj, fp, SEAM_RINGS))
    _set_group(obj, SEAM_NEAR_VG, _grow(obj, fp, 1))
    ref = bpy.data.objects.get(SEAM_REF)
    if ref is None:
        ref = bpy.data.objects.new(SEAM_REF, obj.data)
        obj.users_collection[0].objects.link(ref)
    elif ref.data is not obj.data:
        ref.data = obj.data
    ref.parent = obj
    ref.matrix_parent_inverse.identity()
    ref.matrix_basis.identity()
    for m in list(ref.modifiers):
        ref.modifiers.remove(m)
    mk = ref.modifiers.new("v3_seam_band", 'MASK')
    mk.vertex_group = SEAM_BAND_VG
    sm = C.display_smooth_mod(obj)
    if sm is not None:
        ss = ref.modifiers.new(C.DISPLAY_SMOOTH, 'SUBSURF')
        for pr in sm.bl_rna.properties:
            if not pr.is_readonly and pr.identifier not in ("name", "rna_type"):
                try:
                    setattr(ss, pr.identifier, getattr(sm, pr.identifier))
                except (AttributeError, TypeError):
                    pass
        _drive_smooth(ss, obj)
    ref["v3_note"] = ("显示辅助物体（不显示、不可选、不渲染）：与 CS_Body 共用同一网格，只保留零件周围一圈并做同样的显示平滑，"
                      "给身体和零件的交界提供连续表面的法线。请勿删除或编辑；删除后下次启动会自动重建")
    ref.hide_select = True
    ref.hide_render = True
    ref.hide_viewport = True
    return ref


def _seam_mod(o, ref, vg=""):
    m = o.modifiers.get(SEAM_MOD)
    if m is None:
        m = o.modifiers.new(SEAM_MOD, 'DATA_TRANSFER')
    m.object = ref
    m.use_object_transform = True
    m.use_loop_data = True
    m.data_types_loops = {'CUSTOM_NORMAL'}
    m.loop_mapping = 'NEAREST_POLYNOR'
    m.use_max_distance = True
    m.max_distance = SEAM_MAX_DIST
    m.vertex_group = vg
    m.show_in_editmode = m.show_on_cage = False
    _tail(o)
    return m


def ensure_seam(obj):
    """Border shading (display only).  Body and part each end at the border with only their own faces, so the renderer computes
    one-sided normals there (8-10 deg apart on average at display level 1 -> a visible light/dark line).  Both now take the normals of
    the one continuous smoothed surface from the hidden helper SEAM_REF (same mesh data as the body, so it always matches).
    No vertex moves; measurements / health / fit use the base mesh as before."""
    ref = _seam_ref(obj)
    for po in part_objects().values():
        if po is None:
            continue
        sm = po.modifiers.get(C.DISPLAY_SMOOTH)
        if sm is not None:
            _drive_smooth(sm, obj)
        _seam_mod(po, ref)
    m = _seam_mod(obj, ref, SEAM_NEAR_VG)
    t = obj.modifiers.get(TRIM_MOD)
    m.show_viewport = m.show_render = bool(t is not None and t.show_viewport)
    return ref


def _seam_ok(obj, pos):
    ref = bpy.data.objects.get(SEAM_REF)
    if ref is None or ref.data is not obj.data:
        return False
    m = obj.modifiers.get(SEAM_MOD)
    if m is None or m.object is not ref or SEAM_NEAR_VG not in obj.vertex_groups or SEAM_BAND_VG not in obj.vertex_groups:
        return False
    return all(po is None or (po.modifiers.get(SEAM_MOD) is not None and po.modifiers[SEAM_MOD].object is ref) for po in pos.values())


def _set_trim(me, flags):
    a = me.attributes.get(TRIM_ATTR)
    if a is not None and (a.domain != 'FACE' or a.data_type != 'BOOLEAN'):
        me.attributes.remove(a)
        a = None
    if a is None:
        a = me.attributes.new(TRIM_ATTR, 'BOOLEAN', 'FACE')
    a.data.foreach_set("value", np.asarray(flags, bool))


def ensure_display(obj):
    """files saved before the seamless-border display (v021 as first built) are upgraded at boot: rebuild the part objects with their
    skirt (same vertex ids, shapes come from the body as always; hide state kept) and replace the old pre-smoothing body Mask; add the
    border-normal helper if it is missing.  Display only: no body vertex moves.  Returns True if something was upgraded."""
    pos = part_objects()
    if not any(pos.values()):
        return False
    old_mask = obj.modifiers.get(MASK_MOD)
    if old_mask is None and all(po is None or TRIM_ATTR in po.data.attributes for po in pos.values()):
        if _seam_ok(obj, pos):
            return False
        ensure_seam(obj)
        return True
    hidden = {p: bool(po.hide_get() or po.get("v3_user_hidden", False)) for p, po in pos.items() if po is not None}
    if old_mask is not None:
        obj.modifiers.remove(old_mask)
    create_objects(obj)
    for p, h in hidden.items():
        if h:
            set_visible(obj, p, False)
    return True


def create_objects(obj):
    """create (or rebuild) the 4 part objects + the body display trim.  Body vertices / faces are not deleted."""
    G = geo(obj)
    co = C.raw_coords(obj)
    coll = obj.users_collection[0]
    bm = _mat(*BODY_MAT)
    if not obj.data.materials:
        obj.data.materials.append(bm)
    sm = C.display_smooth_mod(obj)
    for p in PARTS:
        k, s = p.split("_")
        g = G[p]
        name = part_obj_name(p)
        old = bpy.data.objects.get(name)
        if old is not None:
            me_old = old.data
            bpy.data.objects.remove(old, do_unlink=True)
            if me_old.users == 0:
                bpy.data.meshes.remove(me_old)
        faces, fv, trim, vids = _skirt(obj, g)
        lmap = {int(v): i for i, v in enumerate(vids)}
        me = bpy.data.meshes.new(name + "_mesh")
        me.from_pydata([tuple(co[v]) for v in vids], [], [[lmap[int(v)] for v in f] for f in fv])
        me.update()
        _set_trim(me, trim)
        a = me.attributes.new(VID_ATTR, 'INT', 'POINT')
        a.data.foreach_set("value", vids.astype(np.int32))
        a = me.attributes.new("v3_inner", 'INT', 'POINT')
        a.data.foreach_set("value", g["inner_mask"][vids].astype(np.int32))
        a = me.attributes.new(FIT_ATTR, 'FLOAT_VECTOR', 'POINT')
        a.data.foreach_set("vector", co[vids].astype(np.float32).reshape(-1))
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        me.materials.append(_mat(KINDS[k]["mat"], KINDS[k]["color"]))
        po = bpy.data.objects.new(name, me)
        coll.objects.link(po)
        po.matrix_world = obj.matrix_world.copy()
        po.color = KINDS[k]["color"]
        po["v3_part_id"] = p
        po["v3_part_meta"] = json.dumps(dict(
            id=p, kind=k, side=s.upper(), label=("左" if s == "l" else "右") + KINDS[k]["label"] + "零件", body=obj.name,
            zone=KINDS[k]["zone"], n_verts=int(len(g["vids"])), n_faces=int(len(g["faces"])), n_interface_ring=int(len(g["ring"])),
            n_hidden_skirt_faces=int(trim.sum()), skirt="外圈一圈身体面只用于显示平滑时与身体一致，平滑后删除、不显示、不可编辑",
            band_m=KINDS[k]["band"], topology="与身体该区域相同的顶点编号（vertex id 稳定）",
            interface="接口环顶点属于身体，零件只移动环内顶点；过渡带内安装量从 0 渐变到 1"), ensure_ascii=False)
        po.lock_location = po.lock_rotation = po.lock_scale = (True, True, True)   # move a part with its install point / controls
        if sm is not None:
            m = po.modifiers.new(C.DISPLAY_SMOOTH, 'SUBSURF')
            m.levels, m.render_levels = sm.levels, sm.render_levels
            m.show_viewport, m.show_render = sm.show_viewport, sm.show_render
            m.show_in_editmode = False
        _trim_mod(po)
    ensure_seam(obj)
    update_mask(obj)
    return [part_obj_name(p) for p in PARTS]


def part_objects():
    return {p: bpy.data.objects.get(part_obj_name(p)) for p in PARTS}


def visible_parts():
    return [p for p, o in part_objects().items() if o is not None and not o.hide_get() and not o.hide_viewport]


def update_mask(obj):
    """the body does not draw the faces of the visible parts (no double layer); a hidden part shows the body region again.
    The faces are removed AFTER the display smoothing (trim modifier), so body and part end on the same smoothed border.
    The vertex group keeps recording which body vertices are covered by a visible part."""
    G = geo(obj)
    vis = visible_parts()
    vg = obj.vertex_groups.get(MASK_VG) or obj.vertex_groups.new(name=MASK_VG)
    vg.remove(list(range(len(obj.data.vertices))))
    ids = np.concatenate([G[p]["inner"] for p in vis]) if vis else np.array([], int)
    if len(ids):
        vg.add([int(i) for i in ids], 1.0, 'REPLACE')
    flags = np.zeros(len(obj.data.polygons), bool)
    for p in vis:
        flags[G[p]["faces"]] = True
    _set_trim(obj.data, flags)
    m = _trim_mod(obj)
    m.show_viewport = m.show_render = bool(vis)
    sn = obj.modifiers.get(SEAM_MOD)
    if sn is not None:
        sn.show_viewport = sn.show_render = bool(vis)       # nothing trimmed -> nothing to fix (saves the transfer cost)
    return vis


def set_visible(obj, p, on):
    po = part_objects()[p]
    if po is None:
        return
    po.hide_set(not on)
    po.hide_render = not on
    po["v3_user_hidden"] = not on
    update_mask(obj)


def _read_part(po):
    if po.mode == "EDIT":
        po.update_from_editmode()
    me = po.data
    n = len(me.vertices)
    cur = np.empty(n * 3); me.vertices.foreach_get("co", cur)
    fit = np.empty(n * 3, dtype=np.float32); me.attributes[FIT_ATTR].data.foreach_get("vector", fit)
    vid = np.empty(n, dtype=np.int32); me.attributes[VID_ATTR].data.foreach_get("value", vid)
    return cur.reshape(-1, 3), fit.reshape(-1, 3).astype(np.float64), vid


def pending_edits(obj):
    """-> {part: max displacement (m)} of part objects moved by the user since the last fit"""
    out = {}
    for p, po in part_objects().items():
        if po is None or FIT_ATTR not in po.data.attributes:
            continue
        cur, fit, _ = _read_part(po)
        m = float(np.linalg.norm(cur - fit, axis=1).max()) if len(cur) else 0.0
        if m > CAPTURE_TOL:
            out[p] = m
    return out


def capture(obj):
    """turn user edits on the part objects into the parts' local shape deltas.  Interface-ring / outside vertices belong to the
    body: their moves are discarded (reported).  Returns messages."""
    msgs = []
    G = geo(obj)
    for p, po in part_objects().items():
        if po is None or FIT_ATTR not in po.data.attributes or p not in LAST:
            continue
        cur, fit, vid = _read_part(po)
        dv = cur - fit
        mv = np.linalg.norm(dv, axis=1) > CAPTURE_TOL
        if not mv.any():
            continue
        inner = G[p]["inner_mask"][vid]
        o, R, M = LAST[p]
        Minv = np.linalg.inv(M)
        e = PD[p]["e"]
        loc = (dv @ R) @ Minv.T                      # world -> local, undo the install rotation / scale
        n_in = 0
        for i in np.flatnonzero(mv & inner):
            key = str(int(vid[i]))
            e[key] = [float(a + b) for a, b in zip(e.get(key, [0.0, 0.0, 0.0]), loc[i])]
            n_in += 1
        n_out = int((mv & ~inner).sum())
        if n_in:
            PD[p]["source"] = PD[p]["source"].split("＋")[0] + "＋手工编辑"
            msgs.append("%s：采纳手工编辑 %d 个顶点（最大 %.1f mm）" % (part_obj_name(p), n_in, np.linalg.norm(dv[mv & inner], axis=1).max() * 1000))
        if n_out:
            msgs.append("%s：%d 个接口环或外圈顶点属于身体，不能单独移动，已复原" % (part_obj_name(p), n_out))
    if msgs:
        save_data(obj)
    return msgs


def write_parts(obj, co=None):
    """refit every part object from the body coordinates (also resets the fit reference)"""
    co = C.raw_coords(obj) if co is None else co
    for p, po in part_objects().items():
        if po is None or VID_ATTR not in po.data.attributes:
            continue
        if po.mode == "EDIT":
            continue                                  # never overwrite a mesh the user is editing right now
        me = po.data
        vid = np.empty(len(me.vertices), dtype=np.int32); me.attributes[VID_ATTR].data.foreach_get("value", vid)
        x = co[vid].astype(np.float32).reshape(-1)
        me.vertices.foreach_set("co", x)
        me.attributes[FIT_ATTR].data.foreach_set("vector", x)
        me.update()


def clear_edits(obj, p):
    PD[p] = dict(e={}, source="原始（身体自身形状）")
    save_data(obj)


# ------------------------------------------------------------------ replacement sources (same topology)
def _mirror_part(p):
    k, s = p.split("_")
    return k + "_" + ("r" if s == "l" else "l")


def source_path(kind, name):
    return os.path.join(SRC_DIR, "%s_%s.json" % (kind, name))


def list_sources(kind):
    if not os.path.isdir(SRC_DIR):
        return []
    return sorted(f[len(kind) + 1:-5] for f in os.listdir(SRC_DIR) if f.startswith(kind + "_") and f.endswith(".json"))


def save_source(obj, p, name, note=""):
    """save part p's current shape deltas as a replaceable source (stored for the LEFT ids; the right side is mirrored)"""
    os.makedirs(SRC_DIR, exist_ok=True)
    k, s = p.split("_")
    e = PD[p]["e"]
    if s == "r":
        mv = A.P.mvmap
        e = {str(int(mv[int(v)])): x for v, x in e.items()}
    path = source_path(k, name)
    json.dump(dict(kind=k, name=name, note=note, topology_sha256=SC._ctx(obj)["topology_sha256"], frame="local (外侧, 外凸, 上), m",
                   deltas=e), open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return path


def apply_source(obj, p, name, mirror=True):
    """replace part p (and its mirror when mirror=True) by a same-topology source.  Raises ValueError on a topology mismatch."""
    k, _ = p.split("_")
    d = json.load(open(source_path(k, name), encoding="utf-8"))
    if d.get("topology_sha256") != SC._ctx(obj)["topology_sha256"]:
        raise ValueError("零件来源的网格拓扑与当前身体不同：不能直接替换（任意拓扑导入未实现）")
    mv = A.P.mvmap
    G = geo(obj)
    done = []
    for q in ([p, _mirror_part(p)] if mirror else [p]):
        e = d["deltas"] if q.endswith("_l") else {str(int(mv[int(v)])): x for v, x in d["deltas"].items()}
        e = {v: x for v, x in e.items() if G[q]["inner_mask"][int(v)]}
        PD[q] = dict(e={v: list(map(float, x)) for v, x in e.items()}, source="替换：" + name)
        done.append(q)
    save_data(obj)
    return done


# ------------------------------------------------------------------ install (mount) points
def mount_specs(obj):
    co = A.rest_co(obj)
    zone = C.load_zones(obj)
    out = []
    nip = C.group_indices(obj, "nipple")
    for k in KINDS:
        for s in ("l", "r"):
            p = k + "_" + s
            sg = 1.0 if s == "l" else -1.0
            if k == "bust":
                v = nip[co[nip, 0] * sg > 0]
                hint = "胸零件安装点（胸前中心）：拖动=整个胸零件移位（左右/前后/上下，默认左右镜像）；朝向/范围在面板「零件安装」里调"
            else:
                # v3: the part centre (largest install weight), not the glute apex - the apex lies 3.5 cm from the part edge (cleft
                # side) where the install moved it only 13 % of the value
                g = geo(obj)[p]
                _, dist = band_weight(co, g, KINDS[k]["band"])
                w = trans_weight(co, g, dist)
                r = g["inner"]
                c = r[np.argmax(w[r])]
                v = r[np.argsort(np.linalg.norm(co[r] - co[c], axis=1))[:4]]
                hint = "臀零件安装点（臀零件中心）：拖动=整个臀零件移位（默认左右镜像）；朝向/范围在面板「零件安装」里调"
            sp = A._spec("mount_%s_%s" % (k, s), ("左" if s == "l" else "右") + KINDS[k]["label"] + "零件安装点", "零件安装", s.upper(), hint,
                         verts=[int(i) for i in v], main=True, axes="XYZ", local_axes="",
                         owners={"*": [cid(p, "x"), cid(p, "y"), cid(p, "z")]}, region=[KINDS[k]["zone"]], radius=6, kind="surface",
                         source="v3 零件安装")
            sp["rest"] = [round(float(x), 5) for x in co[sp["verts"]].mean(0)]
            sp["mirror"] = "mount_%s_%s" % (k, "r" if s == "l" else "l")
            sp["stage"] = "A3"
            sp["part"] = p
            out.append(sp)
    return out


def add_mount_points(obj):
    reg = json.loads(obj["cs_points"])
    have = {p["id"] for p in reg}
    new = [sp for sp in mount_specs(obj) if sp["id"] not in have]
    reg += new
    obj["cs_points"] = json.dumps(reg)
    A.load(obj)
    return [sp["id"] for sp in new]


def _mirror_install(st, p):
    """copy part p's install values to its mirror (x / yaw are in the mirrored local frame, so the values are equal)"""
    q = _mirror_part(p)
    for ax, *_ in AX:
        st[cid(q, ax)] = st[cid(p, ax)]
    return st


# ------------------------------------------------------------------ welded composite copy (A3: "组合副本必须真正焊接")
COMP = "CS_Body_Composite"
COMP_COLL = "CS_Composite"


def _components(nv, ea, eb):
    """union-find over edges -> component label per vertex"""
    par = np.arange(nv)

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for a, b in zip(ea.tolist(), eb.tolist()):
        ra, rb = root(a), root(b)
        if ra != rb:
            par[ra] = rb
    return np.array([root(i) for i in range(nv)])


def build_composite(obj):
    """ONE new mesh = body faces (helpers dropped) with the current part shape, pink / orange material on the part faces.
    Same vertex count / ids as the body; part faces and body faces share the interface-ring vertices (truly welded, not a
    Ctrl+J join of separate pieces).  A snapshot: rebuild after further changes.  Returns a report dict."""
    co = C.raw_coords(obj)
    zone = C.load_zones(obj)
    ls, lt, lv = _faces(obj)
    helper = zone == C.ZID["helper"]
    all_helper = np.add.reduceat(helper[lv].astype(np.int32), ls) == lt
    keep = np.flatnonzero(~all_helper)
    G = geo(obj)
    mat_idx = np.zeros(len(ls), dtype=np.int32)
    for p in PARTS:
        mat_idx[G[p]["faces"]] = 1 if p.startswith("bust") else 2
    old = bpy.data.objects.get(COMP)
    vis = old is not None and not old.hide_get()
    if old is not None:
        me_old = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if me_old.users == 0:
            bpy.data.meshes.remove(me_old)
    me = bpy.data.meshes.new(COMP + "_mesh")
    me.from_pydata([tuple(v) for v in co], [], [lv[ls[f]:ls[f] + lt[f]].tolist() for f in keep])
    me.update()
    sm = np.empty(len(obj.data.polygons), dtype=bool); obj.data.polygons.foreach_get("use_smooth", sm)
    me.polygons.foreach_set("use_smooth", sm[keep])
    me.polygons.foreach_set("material_index", mat_idx[keep])
    for name, col in (BODY_MAT, (KINDS["bust"]["mat"], KINDS["bust"]["color"]), (KINDS["glute"]["mat"], KINDS["glute"]["color"])):
        me.materials.append(_mat(name, col))
    a = me.attributes.new(VID_ATTR, 'INT', 'POINT')
    a.data.foreach_set("value", np.arange(len(co), dtype=np.int32))
    coll = bpy.data.collections.get(COMP_COLL)
    if coll is None:
        coll = bpy.data.collections.new(COMP_COLL)
        bpy.context.scene.collection.children.link(coll)
    ob = bpy.data.objects.new(COMP, me)
    coll.objects.link(ob)
    ob.matrix_world = obj.matrix_world.copy()
    s0 = C.display_smooth_mod(obj)
    if s0 is not None:
        m = ob.modifiers.new(C.DISPLAY_SMOOTH, 'SUBSURF')
        m.levels, m.render_levels, m.show_viewport, m.show_render = s0.levels, s0.render_levels, s0.show_viewport, s0.show_render
    ob.hide_set(not vis)
    ob.hide_render = not vis
    # weld check: every part's interface ring is used by part faces AND body faces of this one mesh, part + torso one component
    fv = [lv[ls[f]:ls[f] + lt[f]] for f in keep]
    used_part = np.zeros(len(co), bool); used_body = np.zeros(len(co), bool)
    for f, vv in zip(keep, fv):
        (used_part if mat_idx[f] else used_body)[vv] = True
    e = np.array([(int(vv[i]), int(vv[(i + 1) % len(vv)])) for vv in fv for i in range(len(vv))])
    comp = _components(len(co), e[:, 0], e[:, 1])
    torso = np.flatnonzero(zone == C.ZID["ribcage"])[:1]
    rep = dict(object=COMP, n_verts=len(me.vertices), n_faces=len(me.polygons), body_verts=len(obj.data.vertices), parts={})
    for p in PARTS:
        ring = G[p]["ring"]
        rep["parts"][p] = dict(ring=int(len(ring)), ring_shared=int((used_part[ring] & used_body[ring]).sum()),
                               same_component_as_torso=bool(len(torso) and np.all(comp[G[p]["inner"]] == comp[torso[0]])))
    ed = np.sort(e, axis=1)
    key, inv, cnt = np.unique(ed, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    rep["edges_used_by_more_than_2_faces"] = int((cnt > 2).sum())
    # edge-level weld proof: every boundary edge of the part faces is shared by exactly one part face and one body face
    fm = np.repeat(mat_idx[keep], [len(vv) for vv in fv])
    rep["interface_edges"] = {}
    for k, mi in (("bust", 1), ("glute", 2)):
        n_k = np.bincount(inv, weights=(fm == mi).astype(np.float64), minlength=len(key))
        n_b = np.bincount(inv, weights=(fm == 0).astype(np.float64), minlength=len(key))
        rep["interface_edges"][k] = dict(part_body_shared=int(((n_k == 1) & (n_b == 1)).sum()),
                                         part_open=int(((n_k == 1) & (cnt == 1)).sum()))
    rep["welded"] = bool(rep["edges_used_by_more_than_2_faces"] == 0 and all(r["same_component_as_torso"] for r in rep["parts"].values())
                         and all(v["part_open"] == 0 and v["part_body_shared"] > 0 for v in rep["interface_edges"].values()))
    rep["mesh_open_edges_total"] = int((cnt == 1).sum())
    rep["max_vertex_offset_vs_body_mm"] = round(float(np.abs(np.array([v.co[:] for v in me.vertices]) - co).max()) * 1000, 4)
    ob["v3_composite"] = json.dumps(dict(source=obj.name, built=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                                         note="一个网格：零件面与身体面共用接口环顶点（真正焊接，不是 Ctrl+J 拼接）；快照，修改后需重建",
                                         **{k: v for k, v in rep.items() if k != "object"}), ensure_ascii=False)
    return rep


def show_composite(obj, on):
    """on: show the welded copy, hide body + parts (their state is untouched); off: back"""
    ob = bpy.data.objects.get(COMP)
    if ob is None:
        return False
    ob.hide_set(not on); ob.hide_render = not on
    obj.hide_set(on); obj.hide_render = on
    for p, po in part_objects().items():
        if po is not None:
            vis = not on and not po.get("v3_user_hidden", False)
            po.hide_set(not vis); po.hide_render = not vis
    update_mask(obj)
    return True


# ------------------------------------------------------------------ registration
_ORIG = {}


def _apply_state(obj, state):
    if obj.get(PROP) is not None and not _ORIG.get("busy"):
        msgs = capture(obj)
        if msgs:
            _ORIG.setdefault("log", []).extend(msgs)
    full = _ORIG["apply_state"](obj, state)
    if obj.get(PROP) is not None and not _ORIG.get("busy"):
        write_parts(obj)
    return full


def _snapshot():
    s = _ORIG["snapshot"]()
    s["parts"] = snapshot()
    return s


def _restore(obj, snap):
    _ORIG["restore"](obj, snap)
    if "parts" in snap:
        restore(obj, snap["parts"])


def guarded_apply(obj, st_new, st_old=None, h0=None):
    """apply an install-slider change with the health guard: if the result adds a NEW spike / crease / fold, the change is scaled
    back (bisection, 6 steps) to the largest safe fraction.  Returns (state, messages).
    h0 = health result of st_old when already known (the panel keeps it after every commit) -> saves one apply + check."""
    import a_health as H
    st_old = dict(C.get_state(obj) if st_old is None else st_old)
    if h0 is None:
        C.apply_state(obj, st_old)
        h0 = H.check(obj)
    C.apply_state(obj, st_new)
    if not H.new_since(h0, H.check(obj)):
        return dict(st_new), []
    lo, hi, best = 0.0, 1.0, 0.0
    for _ in range(6):
        t = 0.5 * (lo + hi)
        C.apply_state(obj, {k: st_old.get(k, v) + t * (v - st_old.get(k, v)) for k, v in st_new.items()})
        if H.new_since(h0, H.check(obj)):
            hi = t
        else:
            lo = best = t
    st = {k: st_old.get(k, v) + best * (v - st_old.get(k, v)) for k, v in st_new.items()}
    C.apply_state(obj, st)
    return st, ["【零件安装再往这个方向会新增尖刺/折痕，已自动收到安全位置（约 %d%% 的改动量）】" % round(best * 100)]


def _make_drag(Base):
    class MountDrag(Base):
        """install points: owners = the part's own install x/y/z (solved also with the mirror OFF); mirror ON copies the result to
        the other side.  No local offset (local_axes "")."""

        def __init__(self, obj, pid):
            super().__init__(obj, pid)
            if self.sp.get("part"):
                import a_health as H
                self.h0 = H.check(obj)          # v3: install drags are health-guarded like white / red points (safe range is small at the bust fold)
                if not self.groups and self.sp["owners"]:
                    self._jacobian()

        def _update(self, target):
            st, msg = super()._update(target)
            p = self.sp.get("part")
            if p and self.mir_on:
                _mirror_install(st, p)
                C.apply_state(self.obj, st)
                msg.append("左右镜像：另一侧零件同步")
            return st, msg

        def preview(self, target):
            if self.sp.get("part"):
                st, msg = self._update(target)          # exact (the part band / frame are not linear): ~0.2 s per step
                return msg
            return super().preview(target)
    return MountDrag


def register():
    if _ORIG:
        return
    import v3_body as VB
    VB.register()
    for c in CONTROLS:
        if c["id"] not in SC.CUSTOM_IDS:
            SC.CUSTOM.append(c)
            SC.CUSTOM_IDS.append(c["id"])
        if c["id"] not in C.CTRL:
            C.CONTROLS.append(c)
            C.CTRL[c["id"]] = c
    SC._DEFORMERS["v3_parts_on"] = d_v3_parts
    p = list(SC._POST)
    if "v3_parts_on" not in p:
        p.append("v3_parts_on")
    SC._POST = tuple(p)
    _ORIG.update(apply_state=C.apply_state, snapshot=A.snapshot, restore=A.restore)
    C.apply_state = _apply_state
    A.snapshot = _snapshot
    A.restore = _restore
    A.Drag = _make_drag(A.Drag)                      # call after v3_points.register() (V3Drag underneath)
