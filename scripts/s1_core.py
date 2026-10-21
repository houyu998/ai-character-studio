"""Stage-1 (whole-body draft) core: parameter model, application, measurement, zones, rendering helpers.

Imported by s1_build.py / s1_fit.py / s1_panel.py (run inside Blender).  Every state is a plain dict
{control_id: value}; the dict is stored on the basemesh as custom property "cs_state" and is the single source of
truth that can be re-applied to a fresh MPFB human (inheritable body parameters, per the task book).
"""
import os
import json
import math

import bpy
import addon_utils
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERS = os.path.join(ROOT, "versions")
RENDERS = os.path.join(ROOT, "renders")
REFDIR = os.path.join(ROOT, "reference")
MPFB = "bl_ext.user_default.mpfb"

HumanService = TargetService = ModifierService = HumanObjectProperties = None
CUSTOM_HOOK = None          # set by s1_custom.register(): applies the project's own trial deformations after MPFB's


def enable_mpfb():
    global HumanService, TargetService, ModifierService, HumanObjectProperties
    addon_utils.enable(MPFB, default_set=True, persistent=False)
    from bl_ext.user_default.mpfb.services import HumanService as _H, TargetService as _T, ModifierService as _M
    from bl_ext.user_default.mpfb.entities.objectproperties import HumanObjectProperties as _P
    HumanService, TargetService, ModifierService, HumanObjectProperties = _H, _T, _M, _P


def find_basemesh():
    for o in bpy.data.objects:
        if o.type == 'MESH' and HumanObjectProperties.get_value("is_human_project", entity_reference=o):
            return o
    raise RuntimeError("no MPFB basemesh in file")


# ------------------------------------------------------------------ control table
# kind: macro  -> MPFB macro slider (0..1)
#       bi     -> bipolar detail: value in [-1,1]; +v drives the 'incr' target(s), -v the 'decr' target(s)
#       custom -> NEW trial deformation created by this project on a copy (value 0..1)  [not in MPFB]
# 'have' says whether MPFB already ships the control ('已有') or it is a trial addition ('新增').
# Reasonable slider bounds for the draft (MPFB itself allows +-1 / 0..1, which gives inflated bodies).
# Anything beyond is an "over-range" deformation: it is tried on a copy and only enters the preset library after the
# user confirms it.
BI_RANGE = 0.8
MACRO_RANGE = {"height": (0.35, 0.70), "proportions": (0.35, 0.85), "weight": (0.30, 0.70), "muscle": (0.30, 0.75),
               "cupsize": (0.30, 1.00), "firmness": (0.20, 1.00)}


def _bi(id, label, group, base, note="", sides=False, stage="①"):
    names = [base] if not sides else ["l-" + base, "r-" + base]
    return dict(id=id, label=label, group=group, kind="bi", incr=[n + "-incr" for n in names],
                decr=[n + "-decr" for n in names], lo=-BI_RANGE, hi=BI_RANGE, default=0.0, have="已有", note=note, stage=stage)


def _bi2(id, label, group, incr, decr, note="", stage="①"):
    return dict(id=id, label=label, group=group, kind="bi", incr=incr, decr=decr, lo=-BI_RANGE, hi=BI_RANGE, default=0.0,
                have="已有", note=note, stage=stage)


def _macro(id, label, group, key, default=0.5, note="", stage="①"):
    lo, hi = MACRO_RANGE.get(id, (0.0, 1.0))
    return dict(id=id, label=label, group=group, kind="macro", key=key, lo=lo, hi=hi, default=default,
                have="已有", note=note, stage=stage)


CONTROLS = [
    # ---- overall
    _macro("height", "身高（宏）", "整体", "height", note="整体身高与比例联动"),
    _macro("proportions", "比例（宏）", "整体", "proportions", note="MPFB 宏：理想化↔写实比例"),
    _macro("weight", "体重（宏）", "整体", "weight"),
    _macro("muscle", "肌肉（宏）", "整体", "muscle"),
    _bi("head_scale_v", "头高", "头颈", "head-scale-vert", "影响头身比"),
    _bi("head_scale_h", "头宽", "头颈", "head-scale-horiz"),
    _bi("neck_len", "颈长", "头颈", "measure-neck-height"),
    # ---- torso
    _bi("shoulder_w", "肩宽", "上身", "measure-shoulder-dist"),
    _bi("chest_w", "胸廓宽", "上身", "torso-scale-horiz"),
    _bi("chest_d", "胸廓厚", "上身", "torso-scale-depth"),
    _bi("torso_len", "躯干长", "上身", "torso-scale-vert"),
    _bi2("torso_fb", "躯干前后位移", "上身", ["torso-trans-forward"], ["torso-trans-backward"]),
    _bi("underbust_c", "下胸围（胸廓）", "上身", "measure-underbust-circ"),
    _bi("vshape", "肩胸倒三角", "上身", "torso-vshape"),
    _bi("waist_c", "腰围", "腰腹", "measure-waist-circ"),
    _bi("belly", "小腹", "腰腹", "stomach-pregnant"),
    # ---- chest (overall volume only at this stage)
    _macro("cupsize", "胸量（罩杯宏）", "胸", "cupsize", note="宏参数：整体胸部体积"),
    _macro("firmness", "胸紧致（宏）", "胸", "firmness"),
    _bi("bust_c", "胸围", "胸", "measure-bust-circ", note="胸围一并牵动胸廓"),
    # v002: 位置 / 间距 / 上下体积分布是参考图里最明显的"胸大形"差距（胸位高、两胸靠拢有乳沟、上半球饱满），拉进阶段①；
    # 乳头、尖度这类细节仍留在阶段②
    _bi2("bust_pos", "胸根位置（升/降）", "胸", ["breast-trans-up"], ["breast-trans-down"], note="v002 起在阶段①：胸位高低"),
    _bi("bust_dist", "胸间距（负=靠拢）", "胸", "breast-dist", note="v002 起在阶段①：两胸靠拢/乳沟"),
    _bi("bust_point", "胸突出/尖度", "胸", "breast-point", note="阶段②细化", stage="②"),
    _bi2("bust_vol_v", "胸上下体积分布（正=上半球更满）", "胸", ["breast-volume-vert-up"], ["breast-volume-vert-down"], note="v002 起在阶段①：上半球饱满度"),
    # ---- hip / buttocks
    _bi("hip_w", "胯宽", "胯臀", "hip-scale-horiz"),
    _bi("hip_d", "胯厚（前后）", "胯臀", "hip-scale-depth"),
    _bi("hip_h", "胯高（纵向）", "胯臀", "hip-scale-vert"),
    _bi("hip_c", "胯围", "胯臀", "measure-hips-circ"),
    _bi("glute_vol", "臀体积（MPFB 仅此 1 项）", "胯臀", "buttocks-volume", note="臀部唯一现成控制"),
    _bi2("hip_fb", "胯前后位移", "胯臀", ["hip-trans-forward"], ["hip-trans-backward"], note="骨盆整体前/后移（站姿重心）"),
    _bi2("waist_hip_pos", "腰胯过渡高低", "胯臀", ["hip-waist-up"], ["hip-waist-down"]),
    _bi("pelvis_bulge", "下腹/耻骨前凸", "胯臀", "bulge"),
    # ---- legs
    _bi("thigh_len", "大腿长", "腿", "upperlegs-height"),
    _bi("shin_len", "小腿长", "腿", "lowerlegs-height"),
    _bi("thigh_c", "大腿围", "腿", "measure-thigh-circ"),
    _bi("thigh_fat", "大腿丰满", "腿", "upperleg-fat", sides=True),
    _bi("thigh_depth", "大腿前后厚", "腿", "upperleg-scale-depth", sides=True),
    _bi("calf_fat", "小腿丰满", "腿", "lowerleg-fat", sides=True),
    _bi("calf_c", "小腿围", "腿", "measure-calf-circ"),
    # ---- arms
    _bi("uarm_len", "上臂长", "臂", "measure-upperarm-length"),
    _bi("larm_len", "前臂长", "臂", "measure-lowerarm-length"),
    _bi("uarm_c", "上臂围", "臂", "measure-upperarm-circ"),
]
# ---- stage B (MPFB native controls; stage "B"): abdomen / arms / legs / hands / feet detail
CONTROLS += [
    _bi("abd_tone", "腹部 紧实↔柔软", "阶段B·腰腹", "stomach-tone", note="MPFB 腹部松紧", stage="B"),
    _bi2("navel_ud", "肚脐 上↔下", "阶段B·腰腹", ["stomach-navel-up"], ["stomach-navel-down"], stage="B"),
    _bi2("navel_io", "肚脐 凹↔凸", "阶段B·腰腹", ["stomach-navel-out"], ["stomach-navel-in"], stage="B"),
    _bi("shoulder_muscle", "三角肌/肩部肌肉", "阶段B·四肢", "upperarm-shoulder-muscle", sides=True, stage="B"),
    _bi("uarm_fat", "上臂脂肪", "阶段B·四肢", "upperarm-fat", sides=True, stage="B"),
    _bi("uarm_muscle", "上臂肌肉", "阶段B·四肢", "upperarm-muscle", sides=True, stage="B"),
    _bi("larm_fat", "前臂脂肪", "阶段B·四肢", "lowerarm-fat", sides=True, stage="B"),
    _bi("larm_muscle", "前臂肌肉", "阶段B·四肢", "lowerarm-muscle", sides=True, stage="B"),
    _bi("thigh_muscle", "大腿肌肉", "阶段B·四肢", "upperleg-muscle", sides=True, stage="B"),
    _bi("calf_muscle", "小腿肌肉", "阶段B·四肢", "lowerleg-muscle", sides=True, stage="B"),
    _bi("knee_c", "膝围", "阶段B·四肢", "measure-knee-circ", stage="B"),
    _bi("leg_valgus", "腿型 O↔X", "阶段B·四肢", "leg-valgus", sides=True, stage="B"),
    _bi("hand_scale", "手整体大小", "阶段B·手", "hand-scale", sides=True, stage="B"),
    _bi("finger_len", "手指长度", "阶段B·手", "hand-fingers-length", sides=True, stage="B"),
    _bi("finger_dia", "手指粗细", "阶段B·手", "hand-fingers-diameter", sides=True, stage="B"),
    _bi("finger_dist", "手指间距（并拢↔张开）", "阶段B·手", "hand-fingers-distance", sides=True, stage="B"),
    _bi("wrist_c", "腕围", "阶段B·手", "measure-wrist-circ", stage="B"),
    _bi("foot_scale", "脚整体大小", "阶段B·脚", "foot-scale", sides=True, stage="B"),
    _bi("foot_len", "脚长（前后）", "阶段B·脚", "foot-scale-depth", sides=True, stage="B"),
    _bi("foot_w", "脚宽", "阶段B·脚", "foot-scale-horiz", sides=True, stage="B"),
    _bi("foot_h", "脚背高/脚厚", "阶段B·脚", "foot-scale-vert", sides=True, stage="B"),
    _bi("ankle_c", "踝围", "阶段B·脚", "measure-ankle-circ", stage="B"),
]

# stage-B safe ranges: measured with b_range_scan.py (a_health "new issues since rest" = 0 up to this value with AI auto-refine on).
# Values in B_NEEDS_REFINE are only issue-free when refine_auto = 1 (otherwise the scan found new spikes/creases beyond the listed |value|).
B_RANGE = {"abd_tone": (-0.8, 0.6), "navel_ud": (-0.8, 0.8), "navel_io": (-0.4, 0.8), "shoulder_muscle": (-0.8, 0.4), "hand_scale": (-0.6, 0.4),
           "finger_len": (-0.8, 0.6), "finger_dia": (-0.6, 0.6), "leg_valgus": (-0.8, 0.6), "foot_scale": (-0.6, 0.8), "foot_len": (-0.6, 0.8),
           "foot_w": (-0.6, 0.8), "foot_h": (-0.4, 0.8)}
B_NEEDS_REFINE = {"navel_ud": "下", "uarm_fat": "负", "uarm_muscle": "正>0.4", "bust_sag": "上托", "bust_yaw": "内收/外展>0.25", "bust_pitch": "朝下",
                  "bust_lowfull": "收薄/>0.25"}
for _c in CONTROLS:
    if _c["id"] in B_RANGE:
        _c["lo"], _c["hi"] = B_RANGE[_c["id"]]
    if _c["id"] in B_NEEDS_REFINE:
        _c["note"] = (_c.get("note") or "") + "（%s方向较大时需开“AI 自动细化”，否则会出现新尖点/折痕）" % B_NEEDS_REFINE[_c["id"]]
CTRL = {c["id"]: c for c in CONTROLS}
# NOTE: 'custom' trial deformations are appended by s1_custom.py (imported lazily) so that this table stays
# strictly "what MPFB already gives us".


def default_state():
    return {c["id"]: c["default"] for c in CONTROLS}


def clamp_state(state):
    out = {}
    for k, v in state.items():
        c = CTRL.get(k)
        if c is None:
            continue
        if not math.isfinite(float(v)):
            raise ValueError("控制 %s 的值不是有限数（%r）：已拒绝，网格未改动" % (k, v))
        out[k] = float(min(max(v, c["lo"]), c["hi"]))
    return out


# ------------------------------------------------------------------ apply
def _set_key(obj, name, value):
    keys = obj.data.shape_keys
    if keys is not None and name in keys.key_blocks:
        if abs(keys.key_blocks[name].value - value) > 1e-9:
            keys.key_blocks[name].value = value
        return
    if value < 1e-6:
        return
    path = TargetService.target_full_path(name)
    if not path:
        raise RuntimeError("unknown MPFB target: " + name)
    TargetService.load_target(obj, path, weight=float(value), name=name)


def apply_state(obj, state):
    """Apply a control-state dict to the basemesh (bounded). Unlisted controls keep their default."""
    if obj.mode != "OBJECT":                     # MPFB shape-key edits assert OBJECT mode; a file saved in sculpt/edit mode would half-apply
        bpy.ops.object.mode_set(mode="OBJECT")
    full = default_state()
    full.update(clamp_state(state))
    macro_changed = False
    for c in CONTROLS:
        if c["kind"] != "macro":
            continue
        cur = HumanObjectProperties.get_value(c["key"], entity_reference=obj)
        if abs(cur - full[c["id"]]) > 1e-9:
            HumanObjectProperties.set_value(c["key"], float(full[c["id"]]), entity_reference=obj)
            macro_changed = True
    if macro_changed:
        TargetService.reapply_macro_details(obj)
    for c in CONTROLS:
        if c["kind"] != "bi":
            continue
        v = full[c["id"]]
        for n in c["incr"]:
            _set_key(obj, n, max(v, 0.0))
        for n in c["decr"]:
            _set_key(obj, n, max(-v, 0.0))
    if CUSTOM_HOOK is not None:
        CUSTOM_HOOK(obj, full)
    obj["cs_state"] = json.dumps(full)
    return full


def get_state(obj):
    s = default_state()
    raw = obj.get("cs_state")
    if raw:
        s.update(json.loads(raw))
    return s


# ------------------------------------------------------------------ display smoothing
# Blender's built-in Subdivision Surface modifier, used for DISPLAY only: the MPFB base mesh is sparse (10-20 mm spacing
# on the bust), so its polygonal outline / lower-bust "step" shows up in the viewport and in renders.  The modifier does
# not touch the base-mesh vertices: every measurement (Profiler, fit, silhouette overlay, IoU) is done on the base mesh.
DISPLAY_SMOOTH = "cs_display_smooth"
SMOOTH_VIEW_LEVEL = 1
SMOOTH_RENDER_LEVEL = 2


def display_smooth_mod(obj, create=False):
    m = obj.modifiers.get(DISPLAY_SMOOTH)
    if m is None and create:
        m = obj.modifiers.new(DISPLAY_SMOOTH, 'SUBSURF')
        m.levels, m.render_levels = SMOOTH_VIEW_LEVEL, SMOOTH_RENDER_LEVEL
        m.show_in_editmode = False                     # edit mode keeps showing the real base mesh
        m.show_on_cage = False
    return m


def set_display_smooth(obj, on):
    m = display_smooth_mod(obj, create=bool(on))
    if m is not None:
        m.show_viewport = m.show_render = bool(on)
    return m


def is_display_smooth(obj):
    m = obj.modifiers.get(DISPLAY_SMOOTH)
    return bool(m is not None and m.show_viewport)


class base_mesh_only:
    """Context manager: display smoothing off (silhouettes / metrics are taken on the base mesh)."""

    def __init__(self, obj):
        self.obj, self.m, self.saved = obj, obj.modifiers.get(DISPLAY_SMOOTH), None

    def __enter__(self):
        if self.m is not None:
            self.saved = (self.m.show_viewport, self.m.show_render)
            self.m.show_viewport = self.m.show_render = False
        return self

    def __exit__(self, *a):
        if self.m is not None and self.saved is not None:
            self.m.show_viewport, self.m.show_render = self.saved


# ------------------------------------------------------------------ geometry
_RAW_LAST = {}


def raw_coords(obj):
    """Vertex coordinates with shape keys mixed, modifiers off."""
    saved = [(m, m.show_viewport) for m in obj.modifiers]
    try:
        for m, _ in saved:
            m.show_viewport = False
    except AttributeError:
        # Blender forbids writing ID data while a panel is being DRAWN (found in the real-GUI test A14): serve the last coordinates taken outside drawing
        last = _RAW_LAST.get(obj.name)
        if last is not None:
            return last.copy()
        raise
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    arr = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", arr)
    ev.to_mesh_clear()
    for m, v in saved:
        m.show_viewport = v
    out = arr.reshape(-1, 3)
    _RAW_LAST[obj.name] = out.copy()
    return out


def raw_coords_cached(obj):
    """For code that runs while the UI is being DRAWN (panels, draw handlers): never touches modifiers / depsgraph (doing so there raises
    'Writing to ID classes in this context is not allowed' and a 'GPU context already bound' error - found in the real-GUI test A14).
    Returns the coordinates of the last raw_coords() call (every commit / undo / load takes one), else a fresh evaluation."""
    last = _RAW_LAST.get(obj.name)
    return last.copy() if last is not None and len(last) == len(obj.data.vertices) else raw_coords(obj)


_GI_CACHE = {}


def group_indices(obj, name):
    gi = obj.vertex_groups.get(name)
    if gi is None:
        return np.array([], dtype=int)
    key = (obj.name, name, len(obj.data.vertices))
    if not name.startswith("_") and key in _GI_CACHE:
        return _GI_CACHE[key]
    idx = gi.index
    out = np.array([v.index for v in obj.data.vertices if any(g.group == idx for g in v.groups)], dtype=int)
    if not name.startswith("_"):
        _GI_CACHE[key] = out
    return out


# ------------------------------------------------------------------ zones (stored on the mesh; follow the vertices)
ZONES = [  # id, key, 中文, colour (sRGB 0-1)
    (0, "helper", "辅助/隐藏", (0.35, 0.35, 0.35)),
    (1, "head", "头", (0.86, 0.86, 0.92)),
    (2, "neck", "颈", (0.30, 0.80, 0.75)),
    (3, "ribcage", "胸廓/上背", (1.00, 0.82, 0.30)),
    (4, "bust", "胸", (1.00, 0.35, 0.60)),
    (5, "waist", "腰腹", (0.60, 0.88, 0.35)),
    (6, "hip", "胯/骨盆", (1.00, 0.60, 0.20)),
    (7, "glute", "臀", (0.90, 0.20, 0.20)),
    (8, "thigh", "大腿", (0.70, 0.50, 1.00)),
    (9, "calf", "小腿", (0.30, 0.42, 0.92)),
    (10, "foot", "脚", (0.75, 0.75, 0.75)),
    (11, "uarm", "上臂", (0.30, 0.75, 0.95)),
    (12, "larm", "前臂", (0.20, 0.60, 0.85)),
    (13, "hand", "手", (0.60, 0.60, 0.60)),
]
ZID = {z[1]: z[0] for z in ZONES}


def _jc(obj, co, name):
    ids = group_indices(obj, name)
    return co[ids].mean(axis=0)


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t


def _side_pt(obj, co, name):
    """Centre of the +x half of a vertex group (joints exist as small cubes; some groups are symmetric)."""
    ids = group_indices(obj, name)
    P = co[ids]
    if (P[:, 0] > 0.005).any() and (P[:, 0] < -0.005).any():
        P = P[P[:, 0] > 0]
    return P.mean(axis=0)


def landmarks(obj, co):
    """Anatomical landmarks (z heights, nipple point, crotch) read from the current geometry."""
    body = group_indices(obj, "body")
    L = {}
    L["neck_z"] = _jc(obj, co, "joint-neck")[2]
    L["shoulder"] = _jc(obj, co, "joint-l-shoulder")
    L["pelvis"] = _jc(obj, co, "joint-pelvis")
    L["hipjoint"] = _jc(obj, co, "joint-l-upper-leg")
    L["knee_z"] = _jc(obj, co, "joint-l-knee")[2]
    L["ankle_z"] = _jc(obj, co, "joint-r-ankle")[2]
    L["nipple"] = _side_pt(obj, co, "nipple")
    L["spine2"] = _jc(obj, co, "joint-spine-2")
    z = co[:, 2]
    mid = np.zeros(len(co), bool)
    mid[body] = True
    mid &= (np.abs(co[:, 0]) < 0.012) & (z > 0.35 * z[body].max()) & (z < 0.62 * z[body].max())
    L["crotch_z"] = float(z[mid].min())
    L["zmin"] = float(z[body].min())
    L["zmax"] = float(z[body].max())
    return L


def compute_zones(obj, co):
    """Zone id per vertex from the neutral-pose geometry (call once, on the base state).
    Bands are placed from anatomical landmarks (nipple, crotch, pelvis, knee ...), not from fixed heights."""
    n = len(co)
    zone = np.zeros(n, dtype=np.int32)
    body = group_indices(obj, "body")
    L = landmarks(obj, co)
    z = co[:, 2]
    ax = np.abs(co[:, 0])
    y = co[:, 1]
    sh = _jc(obj, co, "joint-l-shoulder")
    el = _jc(obj, co, "joint-l-elbow")
    wr = _jc(obj, co, "joint-l-hand")
    fingers = np.mean([_jc(obj, co, "joint-l-finger-3-4"), _jc(obj, co, "joint-l-finger-2-4")], axis=0)

    def mir(v):
        return np.array([abs(v[0]), v[1], v[2]])
    shm, elm, wrm, fgm = map(mir, (sh, el, wr, fingers))
    Pm = np.column_stack([ax, y, z])
    d_ua, _ = _seg_dist(Pm, shm, elm)
    d_la, _ = _seg_dist(Pm, elm, wrm)
    d_ha, _ = _seg_dist(Pm, wrm, fgm)
    d_arm = np.minimum(np.minimum(d_ua, d_la), d_ha)
    shoulder_z = sh[2]
    is_arm = (d_arm < 0.065) & ((ax > 0.14) | (z > shoulder_z - 0.02)) & (z < shoulder_z + 0.05) & (ax > 0.10)
    is_arm |= (d_ha < 0.10) & (ax > 0.30)
    zbody = np.zeros(n, dtype=bool)
    zbody[body] = True
    sel = zbody & ~is_arm
    knee_z, ankle_z = L["knee_z"], L["ankle_z"]
    crotch_z, pel_z = L["crotch_z"], L["pelvis"][2]
    nip = L["nipple"]
    zone[body] = ZID["ribcage"]
    head_z = L["neck_z"] + 0.055
    zone[sel & (z > head_z)] = ZID["head"]
    zone[sel & (z <= head_z) & (z > shoulder_z - 0.025) & (ax < 0.10)] = ZID["neck"]
    torso = sel & (z <= shoulder_z - 0.025) & (z > crotch_z)
    # waist/belly: from the pelvis-top line up to the under-bust line
    under_z = nip[2] - 0.085
    hip_top = pel_z + 0.075
    zone[torso & (z > hip_top) & (z <= under_z)] = ZID["waist"]
    # bust: front cap = ellipsoid around each nipple, front-facing
    mp = np.array([abs(nip[0]), nip[1], nip[2]])
    q = np.column_stack([(ax - mp[0] * 0.92) / 0.105, (y - (mp[1] + 0.05)) / 0.16, (z - (mp[2] + 0.008)) / 0.105])
    is_bust = (np.linalg.norm(q, axis=1) < 1.0) & (y < L["spine2"][1] - 0.02) & torso & (z > under_z - 0.02)
    zone[is_bust] = ZID["bust"]
    # hip/pelvis band + glutes
    hip_sel = sel & (z <= hip_top) & (z > crotch_z - 0.005)
    zone[hip_sel] = ZID["hip"]
    back = y > L["pelvis"][1] + 0.005
    glute = sel & back & (ax < 0.20) & (z <= pel_z + 0.045) & (z > crotch_z - 0.075)
    zone[glute] = ZID["glute"]
    zone[sel & (z <= crotch_z - 0.005) & (z > knee_z) & (zone != ZID["glute"])] = ZID["thigh"]
    zone[sel & (z <= knee_z) & (z > ankle_z + 0.03)] = ZID["calf"]
    zone[sel & (z <= ankle_z + 0.03)] = ZID["foot"]
    uarm = zbody & is_arm & (d_ua <= d_la) & (d_ua <= d_ha)
    larm = zbody & is_arm & (d_la < d_ua) & (d_la <= d_ha)
    hand = zbody & is_arm & (d_ha < d_ua) & (d_ha < d_la)
    zone[uarm] = ZID["uarm"]
    zone[larm] = ZID["larm"]
    zone[hand] = ZID["hand"]
    zone[~zbody] = 0
    return zone


def _adjacency(obj):
    me = obj.data
    e = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def store_zones(obj, zone, smooth=2):
    """zone_id = hard per-vertex id; zone_color = the same colours softened over the mesh graph (display only)."""
    me = obj.data
    for name in ("zone_id", "zone_color"):
        if name in me.attributes:
            me.attributes.remove(me.attributes[name])
    a = me.attributes.new("zone_id", 'INT', 'POINT')
    a.data.foreach_set("value", zone.astype(np.int32))
    col = np.array([z[3] + (1.0,) for z in ZONES], dtype=np.float32)
    c = col[zone].astype(np.float64)
    if smooth:
        ed = _adjacency(obj)
        ok = (zone[ed[:, 0]] != 0) & (zone[ed[:, 1]] != 0)
        ed = ed[ok]
        for _ in range(smooth):
            acc = c.copy()
            cnt = np.ones(len(c))
            np.add.at(acc, ed[:, 0], c[ed[:, 1]])
            np.add.at(acc, ed[:, 1], c[ed[:, 0]])
            np.add.at(cnt, ed[:, 0], 1)
            np.add.at(cnt, ed[:, 1], 1)
            c = acc / cnt[:, None]
    ca = me.attributes.new("zone_color", 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set("color", c.astype(np.float32).reshape(-1))
    me.update()


def load_zones(obj):
    a = obj.data.attributes.get("zone_id")
    if a is None:
        return None
    arr = np.empty(len(obj.data.vertices), dtype=np.int32)
    a.data.foreach_get("value", arr)
    return arr


# ------------------------------------------------------------------ measurement (same definition as the reference)
def _load_ref(name):
    """Reference measurements are made from your own reference image and are not shipped; None if absent."""
    p = os.path.join(REFDIR, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


REF_METRICS = _load_ref("ref01_metrics.json")
REF_LEG = _load_ref("ref01_legs.json")   # per-leg widths (stance-free)
FLS = sorted(float(k) for k in REF_METRICS["profile"].keys()) if REF_METRICS else []


class Profiler:
    """Silhouette numbers at normalised heights, measured exactly like the reference table (fractions of H):
    front_w (full width), leg_w (width of ONE leg = stance-free), side_fwd / side_back (extent front/back of the
    ankle plumb line).  H = sole to top of skull (the reference H includes hair, ~1% taller).
    Uses exact plane/edge intersections (robust to vertex spacing)."""

    def __init__(self, obj, zone, exclude=("uarm", "larm", "hand", "helper"), fls=None):
        self.zone = zone
        self.keep = ~np.isin(zone, [ZID[k] for k in exclude])
        ed = _adjacency(obj)
        ok = self.keep[ed[:, 0]] & self.keep[ed[:, 1]]
        self.ea, self.eb = ed[ok, 0], ed[ok, 1]
        self.fls = fls or FLS

    def slice(self, co, z0):
        za, zb = co[self.ea, 2], co[self.eb, 2]
        m = ((za - z0) * (zb - z0) <= 0) & (za != zb)
        a, b = self.ea[m], self.eb[m]
        t = (z0 - co[a, 2]) / (co[b, 2] - co[a, 2])
        return co[a, :2] + t[:, None] * (co[b, :2] - co[a, :2])

    def __call__(self, co):
        zk = co[self.keep, 2]
        zmin, zmax = float(zk.min()), float(zk.max())
        H = zmax - zmin
        ys = []
        for f in np.linspace(0.085, 0.100, 4):
            q = self.slice(co, zmin + f * H)
            ys.append(0.5 * (q[:, 1].min() + q[:, 1].max()))
        plumb = float(np.mean(ys))
        prof = {}
        for fl in self.fls:
            q = self.slice(co, zmin + fl * H)
            if len(q) < 4:
                continue
            r = q[q[:, 0] > 0]
            inner = float(r[:, 0].min()) if len(r) and (q[:, 0] < 0).any() else 0.0
            prof[fl] = dict(front_w=float(q[:, 0].max() - q[:, 0].min()) / H,
                            out_half=float(q[:, 0].max()) / H, in_half=inner / H,
                            leg_w=float(q[:, 0].max() - inner) / H,
                            side_fwd=float(plumb - q[:, 1].min()) / H,
                            side_back=float(q[:, 1].max() - plumb) / H)
        return dict(H=H, zmin=zmin, plumb=plumb, profile=prof)


def model_profile(obj, co, zone, **kw):
    return Profiler(obj, zone, **kw)(co)


REF_OVERRIDE = {"profile":{}, "legs":{}, "landmarks":{}}          # stage A: {"profile":{fl:{q:v}}, "legs":{fl:{q:v}}, "landmarks":{name:fl}} from user reference images


def ref_value(fl, q):
    """Reference value for quantity q at height fl (leg_w comes from the per-leg table).  None = not known from the
    current reference (only possible with REF_OVERRIDE)."""
    key = "%g" % fl
    if REF_OVERRIDE is not None:
        tab = REF_OVERRIDE["legs"] if q in ("leg_w", "out_half", "in_half") else REF_OVERRIDE["profile"]
        return tab.get(key, {}).get(q)
    if q in ("leg_w", "out_half", "in_half"):
        return REF_LEG[key][q]
    return REF_METRICS["profile"][key][q]


# ------------------------------------------------------------------ rendering (Workbench: clay grey / zone colours)
def _look_at(cam, target):
    d = Vector(target) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def setup_render(res=(800, 1200)):
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.view_settings.view_transform = 'Standard'
    sc.display.render_aa = '8'
    sh = sc.display.shading
    sh.light = 'STUDIO'
    # 'basic.sl' lights left/right symmetrically (measured: mirror-image error 0.006 vs 0.063 for 'rim.sl'), so a symmetric
    # body does not look lopsided in front/back renders
    sh.studio_light = 'basic.sl' if 'basic.sl' in [l.name for l in bpy.context.preferences.studio_lights] else sh.studio_light
    sh.show_cavity = True
    sh.cavity_type = 'BOTH'
    sh.cavity_ridge_factor = 1.0
    sh.cavity_valley_factor = 1.0
    sh.show_object_outline = False
    w = sc.world or bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = False
    w.color = (0.30, 0.30, 0.30)
    return sc


def render_view(obj, path, view="front", zones=False, ortho=True, res=(800, 1200), hide_arms=False, silhouette=False,
                fit_h=None, floor=None, center_x=None, ortho_scale=None, center_z=None):
    """Render one view of the basemesh. view: front|side|back|q34|q34b.
    center_x = lateral camera offset along the image-right axis (front +x, side -y, back -x)."""
    sc = setup_render(res)
    sh = sc.display.shading
    if silhouette:
        sh.light = 'FLAT'
        sh.color_type = 'SINGLE'
        sh.single_color = (1, 1, 1)
        sh.show_cavity = False
        sc.world.color = (0, 0, 0)
    elif zones:
        sh.color_type = 'VERTEX'
        sh.light = 'STUDIO'
        sh.show_cavity = True
    else:
        sh.color_type = 'SINGLE'
        sh.single_color = (0.66, 0.66, 0.66)
    co = raw_coords(obj)
    bodyz = load_zones(obj)
    keep = bodyz != 0 if bodyz is not None else np.ones(len(co), bool)
    zmin, zmax = co[keep, 2].min(), co[keep, 2].max()
    H = zmax - zmin
    cx = 0.0 if center_x is None else center_x
    cz = center_z if center_z is not None else zmin + H * 0.5
    if ortho_scale is None:
        ortho_scale = H * 1.10 * max(1.0, res[0] / res[1] * 1.0) if False else H * 1.10
    cam_data = bpy.data.cameras.new("cs_cam")
    cam = bpy.data.objects.new("cs_cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    sc.camera = cam
    D = 6.0
    az = {"front": 0.0, "side": -90.0, "back": 180.0, "q34": -40.0, "q34b": 140.0}[view]
    a = math.radians(az)
    cam.location = (math.sin(a) * D + math.cos(a) * cx, -math.cos(a) * D + math.sin(a) * cx, cz)
    if ortho:
        cam_data.type = 'ORTHO'
        cam_data.ortho_scale = ortho_scale
    else:
        cam_data.lens = 85
    _look_at(cam, (math.cos(a) * cx, math.sin(a) * cx, cz))
    cam_data.clip_start, cam_data.clip_end = 0.1, 50
    # hide helpers / arms via a temporary mask modifier when asked
    tmp = None
    if hide_arms and bodyz is not None:
        vg = obj.vertex_groups.new(name="_cs_noarm")
        # hide arm-zone vertices outside the torso half-width (a plane cut; the hard zone border is jagged at the shoulder)
        torso = np.isin(bodyz, [ZID["ribcage"], ZID["bust"], ZID["waist"], ZID["hip"], ZID["glute"]])
        x_cut = 1.10 * float(np.abs(co[torso, 0]).max())
        armz = np.isin(bodyz, [ZID["uarm"], ZID["larm"], ZID["hand"]])
        high = co[:, 2] > zmin + 0.52 * (zmax - zmin)          # above the crotch: cut every vertex by plane (clean edge)
        arm = (np.abs(co[:, 0]) > x_cut) & (armz | high)
        keepids = np.where(~arm & (bodyz != 0))[0]
        vg.add([int(i) for i in keepids], 1.0, 'REPLACE')
        tmp = obj.modifiers.new("_cs_mask", 'MASK')
        tmp.vertex_group = "_cs_noarm"
        obj.modifiers.move(len(obj.modifiers) - 1, 0)        # must sit BEFORE the display Subsurf (else the cut is blurred)
    # colour attribute for viewport/workbench
    ca = obj.data.color_attributes.get("zone_color")
    if ca is not None:
        obj.data.color_attributes.active_color = ca
        obj.data.color_attributes.render_color_index = list(obj.data.color_attributes).index(ca)
    sc.render.filepath = path
    if silhouette:
        with base_mesh_only(obj):
            bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(write_still=True)
    if tmp is not None:
        obj.modifiers.remove(tmp)
        obj.vertex_groups.remove(obj.vertex_groups["_cs_noarm"])
    bpy.data.objects.remove(cam, do_unlink=True)
    bpy.data.cameras.remove(cam_data)
