"""v3 (阶段A增强版) body-structure fixes, registered ON TOP of s1_custom (old controls / old versions unchanged).

Diagnosis (A3-01, scripts/v3_diag_layers.py on v020, ablation one parameter at a time, % of body height):
  * the WIDE / THICK back is created in the MPFB-parameter layer, not by our smoothing:
      bust_c 0.8 (measure-bust-circ)  back width +1.99 %H, back depth +2.08 %H  (a circumference target: it inflates the
                                       whole rib ring, back included, and pushes the breasts out sideways past the back)
      vshape 0.6 (torso-vshape)       back width +1.56 %H, shoulder-joint distance +1.94 %H
      chest_w 0.25                    back width +0.82 %H
  * the back STRUCTURE (scapula / spine relief) is lost in two steps: the same MPFB layer turns the back into a broad
    plate, then back_slim / back_anatomy (analytic blobs) and back_relax (Laplacian) smear what is left.
Fix = the responsible layers:
  v3_bustc_back  cancels the bust_c displacement everywhere EXCEPT the front breast mound (the breast volume stays;
                 the rib ring / back / lateral chest go back to the un-inflated shape)
  v3_back_narrow cancels the vshape / chest_w widening on the BACK below the shoulder top (acromion width kept)
  v3_relief      (POST) restores the NATIVE MPFB surface relief (scapula, spine furrow, lower ribs / abdomen, patella,
                 hamstring tendons, olecranon) where the parameter layers flattened it: per region, the normal
                 component of (surface - low-pass surface) is set back to the native value (scaled to the body height,
                 attenuated with body fat).  It never touches the bust / glute zones (those are the independent parts).
The native relief is baked once per mesh into the attribute 'cs_native_relief' (bake_native)."""
import json
import numpy as np
import bpy

import s1_core as C
import s1_custom as SC

RELIEF_ITERS = 8            # low-pass passes that define "relief" (same as the A3-01 metric)
RELIEF_CLAMP_M = 0.008      # max total correction per vertex
RELIEF_STEP_M = 0.003       # max correction per pass
RELIEF_PASSES = 4
RELIEF_HP_ITERS = 8         # each pass keeps only the LOCAL part of the correction (minus its 8-pass low-pass): without it the
                            # fixed-point pass pushed whole sections outwards (v3_exp_relief_hp.py on v020 + X2: waist girth +43 mm,
                            # flanks +6.5 mm each side = the waist_c narrowing undone); with 8: waist +8.8 mm, same back / knee /
                            # elbow relief by eye (work/v3/relief_img), no new health warning
NAT_ATTR = "cs_native_relief"

V3_CONTROLS = [
    dict(id="v3_bustc_back", label="胸围不撑宽后背（v3）", group="v3·身体结构", kind="custom", lo=0.0, hi=1.0, default=0.0,
         have="新增·v3", stage="A3",
         note="v3：MPFB 胸围目标会把整圈胸廓连后背一起撑大；本项只保留胸丘本身的增量，后背/侧胸/肋廓恢复。1=完全去掉"),
    dict(id="v3_back_narrow", label="倒三角不加宽后背（v3）", group="v3·身体结构", kind="custom", lo=0.0, hi=1.0, default=0.0,
         have="新增·v3", stage="A3",
         note="v3：去掉倒三角/胸廓宽参数在肩峰以下后背上的加宽，肩峰宽度保留（肩宽单独由“肩宽”控制）"),
    dict(id="v3_relief", label="身体结构起伏恢复（背/腹/膝/肘）", group="v3·身体结构", kind="custom", lo=0.0, hi=1.5, default=0.0,
         have="新增·v3", stage="A3",
         note="v3：把被参数层抹平的原生结构起伏(肩胛、脊柱沟、下肋/腹、髌骨、腘绳肌腱、肘尖)恢复回来；1=原生强度；胖时自动减弱；胸臀区不动"),
    dict(id="v3_fatdist", label="脂肪分布强度（由胖瘦自动设置）", group="v3·身体结构", kind="custom", lo=-1.0, hi=1.0, default=0.0,
         have="新增·v3", stage="A3",
         note="v3：由“胖瘦”自动设置。负=极瘦（胖瘦<0），正=肥胖（胖瘦>0.5）。叠加 MPFB 原生的局部减脂/增脂目标(大腿/小腿/上臂/小腹/腰/胯)，"
              "权重都在MPFB自身0~1范围内，不做线性外推；胸臀零件随之重新贴合"),
]

# ---- 胖瘦 (A3): user value f, regular 0..1, extended down to -0.5 (extreme thin)
FAT_LO, FAT_HI = -0.5, 1.0
W_AT_0, W_AT_HALF, W_AT_1 = 0.15, 0.50, 1.00        # MPFB weight macro at f = 0 / 0.5 / 1 (MPFB native range is 0..1)
LEAN_TARGETS = (("thigh_fat", 1.0), ("calf_fat", 0.7), ("uarm_fat", 1.0), ("belly", 0.8), ("waist_c", 0.5), ("hip_c", 0.35))
# v3 test (v3_exp_types.py): belly 0.45 + waist_c 1.0 together fold the skin under the bust at 胖瘦 1 (crease 139 deg) and leave a
# bump at the navel; belly 0.25 + waist_c 0.6 removes both (max new crease 74 deg) - cost: waist grows less (fat 1: 708 vs 772 mm)
PLUMP_TARGETS = (("thigh_fat", 0.5), ("calf_fat", 0.5), ("uarm_fat", 1.0), ("belly", 0.25), ("waist_c", 0.6), ("hip_c", 0.45))


def fat_to_state(f):
    """胖瘦 f -> the values that really take effect: MPFB weight macro + v3_fatdist.  Never clamps silently: out-of-range raises."""
    f = float(f)
    if not (FAT_LO - 1e-9 <= f <= FAT_HI + 1e-9):
        raise ValueError("胖瘦 %.3f 超出 %.1f~%.1f" % (f, FAT_LO, FAT_HI))
    if f < 0:
        return dict(weight=W_AT_0 * (1.0 + 2.0 * f), v3_fatdist=2.0 * f)
    if f <= 0.5:
        return dict(weight=W_AT_0 + (W_AT_HALF - W_AT_0) * f / 0.5, v3_fatdist=0.0)
    return dict(weight=W_AT_HALF + (W_AT_1 - W_AT_HALF) * (f - 0.5) / 0.5, v3_fatdist=(f - 0.5) / 0.5)


def state_to_fat(st):
    """inverse of fat_to_state (for a state set some other way, e.g. old versions: weight only)"""
    w = float(st.get("weight", 0.5)); fd = float(st.get("v3_fatdist", 0.0))
    if fd < -1e-6:
        return 0.5 * fd
    if w <= W_AT_HALF:
        return (w - W_AT_0) / (W_AT_HALF - W_AT_0) * 0.5          # may be < 0 for an old weight < 0.15 (shown, not clamped)
    return 0.5 + (w - W_AT_HALF) / (W_AT_1 - W_AT_HALF) * 0.5


def fat_effective_text(st):
    f = state_to_fat(st)
    t = "胖瘦 %.2f → MPFB体重宏 %.2f" % (f, float(st.get("weight", 0.5)))
    fd = float(st.get("v3_fatdist", 0.0))
    if abs(fd) > 1e-6:
        t += " + %s %.2f（MPFB原生局部%s目标）" % ("极瘦形变" if fd < 0 else "增脂分布", abs(fd), "减脂" if fd < 0 else "增脂")
    return t


def _keydata(obj, name):
    kb = obj.data.shape_keys.key_blocks.get(name) if obj.data.shape_keys else None
    if kb is None:
        return None
    a = np.empty(len(kb.data) * 3, dtype=np.float64)
    kb.data.foreach_get("co", a)
    b = np.empty(len(kb.data) * 3, dtype=np.float64)
    kb.relative_key.data.foreach_get("co", b)
    return (a - b).reshape(-1, 3)


_KD = {}


def target_delta(obj, cid, value):
    """displacement that the bi control `cid` at `value` adds (MPFB shape key data, relative to its basis)"""
    c = C.CTRL[cid]
    names = c["incr"] if value > 0 else c["decr"]
    out = None
    for n in names:
        k = (obj.name, n, len(obj.data.vertices))
        if k not in _KD or _KD[k] is None:
            if obj.data.shape_keys is None or n not in obj.data.shape_keys.key_blocks:
                path = C.TargetService.target_full_path(n)        # MPFB target not loaded yet (never used): load at weight 0
                if path:
                    C.TargetService.load_target(obj, path, weight=0.0, name=n)
            _KD[k] = _keydata(obj, n)
        dd = _KD[k]
        if dd is None:
            continue
        out = dd * abs(value) if out is None else out + dd * abs(value)
    return out


_FULL = {}
_ORIG_APPLY = []


def _state(obj):
    """the state being applied right now (apply_state writes cs_state only AFTER the custom hook)"""
    return _FULL.get(obj.name) or C.get_state(obj)


def hook(obj, full):
    _FULL[obj.name] = dict(full)
    try:
        return _ORIG_APPLY[0](obj, full)
    finally:
        _FULL.pop(obj.name, None)


def _ease(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _obj(ctx):
    return bpy.data.objects[ctx["obj_name"]]


def _front_bust_w(co, ctx):
    w = ctx.get("_v3_fbw")
    if w is None:
        z = ctx["zone"]
        b = (z == C.ZID["bust"]).astype(np.float64)
        w = np.clip(SC._smooth(b, ctx["ea"], ctx["eb"], 3), 0.0, 1.0)
        ctx["_v3_fbw"] = w
    return w


def d_v3_bustc_back(co, ctx, amount):
    d = np.zeros_like(co)
    obj = _obj(ctx)
    v = float(_state(obj).get("bust_c", 0.0))
    if amount <= 0 or abs(v) < 1e-6:
        return d
    td = target_delta(obj, "bust_c", v)
    if td is None:
        return d
    keep = _front_bust_w(co, ctx)
    sh = co[ctx["shoulder_ids"]].mean(axis=0)
    # breast mound = bust zone AND in front of the shoulder plane; lateral / back parts of the bust zone are cancelled too
    front = _ease((sh[1] - co[:, 1] + 0.02) / 0.06)
    keep = keep * front
    w = 1.0 - keep
    w[ctx["zone"] == C.ZID["helper"]] = 0.0
    return -td * (w * min(amount, 1.0))[:, None]


def d_v3_back_narrow(co, ctx, amount):
    d = np.zeros_like(co)
    obj = _obj(ctx)
    st = _state(obj)
    if amount <= 0:
        return d
    sh = co[ctx["shoulder_ids"]].mean(axis=0)
    z = ctx["zone"]
    torso = np.isin(z, [C.ZID["ribcage"], C.ZID["waist"], C.ZID["bust"]]).astype(np.float64)
    torso = np.clip(SC._smooth(torso, ctx["ea"], ctx["eb"], 3), 0.0, 1.0)
    w = torso * _ease((co[:, 1] - sh[1] + 0.01) / 0.05) * _ease((sh[2] - 0.03 - co[:, 2]) / 0.06)
    w = np.clip(SC._smooth(w, ctx["ea"], ctx["eb"], 3), 0.0, 1.0)
    w[z == C.ZID["helper"]] = 0.0
    for cid in ("vshape", "chest_w"):
        v = float(st.get(cid, 0.0))
        if v <= 1e-6:
            continue
        td = target_delta(obj, cid, v)
        if td is not None:
            d -= td * (w * min(amount, 1.0))[:, None]
    return d


def _normals(co, ctx):
    tri = ctx.get("_tri")
    if tri is None:
        tri = ctx["_tri"] = SC._fan_tris(ctx)
    fn = np.cross(co[tri[:, 1]] - co[tri[:, 0]], co[tri[:, 2]] - co[tri[:, 0]])
    vn = np.zeros_like(co)
    for j in range(3):
        SC._addat(vn, tri[:, j], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def relief(co, ctx):
    return ((co - SC._laplace(co, ctx["ea"], ctx["eb"], RELIEF_ITERS)) * _normals(co, ctx)).sum(axis=1)


def relief_mask(co, ctx):
    """region weights: back torso, abdomen, knees, elbows (bust / glute / head / hands / feet / helpers = 0)"""
    z = ctx["zone"]
    obj = _obj(ctx)
    sh = co[ctx["shoulder_ids"]].mean(axis=0)
    m = np.zeros(len(co))
    torso = np.isin(z, [C.ZID["ribcage"], C.ZID["waist"], C.ZID["neck"]])
    m[torso & (co[:, 1] > sh[1] - 0.03)] = 1.0                       # back
    m[(z == C.ZID["waist"])] = 1.0                                     # abdomen + flanks
    for s in ("l", "r"):
        for j, r in (("knee", 0.11), ("elbow", 0.08)):
            ids = C.group_indices(obj, "joint-%s-%s" % (s, j))
            if len(ids):
                p = co[ids].mean(axis=0)
                m = np.maximum(m, _ease((r - np.linalg.norm(co - p, axis=1)) / 0.04))
    m[np.isin(z, [C.ZID["bust"], C.ZID["glute"], C.ZID["head"], C.ZID["hand"], C.ZID["foot"], C.ZID["helper"]])] = 0.0
    m = np.clip(SC._smooth(m, ctx["ea"], ctx["eb"], 4), 0.0, 1.0)
    # keep clear of the bust / glute parts and their folds (inframammary fold, gluteal fold): the native (small-bust) relief
    # there is not valid for this body - tried without: 8 new creases along the inframammary fold
    near = ctx.get("_v3_near")
    if near is None:
        pg = np.isin(z, [C.ZID["bust"], C.ZID["glute"]]).astype(np.float64)
        near = ctx["_v3_near"] = np.clip(SC._smooth(pg, ctx["ea"], ctx["eb"], 10) * 4.0, 0.0, 1.0)
    m *= 1.0 - near
    m[np.isin(z, [C.ZID["bust"], C.ZID["glute"], C.ZID["helper"]])] = 0.0
    return m


def _parts_w(ctx):
    w = ctx.get("_v3_partw")
    if w is None:
        z = ctx["zone"]
        pg = np.isin(z, [C.ZID["bust"], C.ZID["glute"]]).astype(np.float64)
        w = ctx["_v3_partw"] = np.clip(SC._smooth(pg, ctx["ea"], ctx["eb"], 4), 0.0, 1.0)
    return w


def d_v3_fatdist(co, ctx, amount):
    """胖瘦 beyond the weight macro: MPFB's own local fat targets (native data, weight <= 1).
    amount < 0: decrease targets (extreme thin); amount > 0: increase targets (obese distribution)"""
    d = np.zeros_like(co)
    if abs(amount) < 1e-9:
        return d
    obj = _obj(ctx)
    a = float(np.clip(amount, -1.0, 1.0))
    st = _state(obj)
    for cid, k in (LEAN_TARGETS if a < 0 else PLUMP_TARGETS):
        if cid not in C.CTRL:
            continue
        v = a * k
        u = float(st.get(cid, 0.0))
        if u * v > 0:                       # same MPFB target as the user's own slider: total weight of that target <= 1 (no extrapolation)
            v = np.sign(v) * min(abs(v), max(0.0, 1.0 - abs(u)))
        if abs(v) < 1e-9:
            continue
        td = target_delta(obj, cid, v)
        if td is not None:
            d += td
    # no parts mask: the bust / glute parts are cut from this body AFTER the deformation, so they follow the fat change
    # (tried masking them: a hard step at the glute border in side view at 胖瘦 0.75 / 1.0)
    d[ctx["zone"] == C.ZID["helper"]] = 0.0
    return d


def native_relief(obj):
    a = obj.data.attributes.get(NAT_ATTR)
    if a is None:
        return None, None
    r = np.empty(len(obj.data.vertices), dtype=np.float64)
    a.data.foreach_get("value", r)
    return r, float(obj.get("cs_native_H", 1.0))


def d_v3_relief(co, ctx, amount):
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    obj = _obj(ctx)
    r0, h0 = native_relief(obj)
    if r0 is None:
        return d
    H = float(np.ptp(co[ctx["body"], 2]))
    wgt = float(_state(obj).get("weight", 0.5))
    fat = 1.0 - 0.6 * float(np.clip((wgt - 0.5) / 0.2, 0.0, 1.0))     # fat covers relief (weight macro 0.5 -> 0.7)
    fd = float(_state(obj).get("v3_fatdist", 0.0))
    fat *= (1.0 - 0.5 * fd) if fd < 0 else (1.0 - 0.6 * fd)             # extreme thin: bony landmarks show more; obese: less
    tgt = r0 * (H / h0) * amount * fat
    m = relief_mask(co, ctx)
    x = co.copy()
    for _ in range(RELIEF_PASSES):                       # the low-pass reference moves with the correction: a few fixed-point passes
        corr = np.clip(tgt - relief(x, ctx), -RELIEF_STEP_M, RELIEF_STEP_M) * m
        if RELIEF_HP_ITERS:
            corr = (corr - SC._smooth(corr, ctx["ea"], ctx["eb"], RELIEF_HP_ITERS)) * m
        x = x + _normals(x, ctx) * corr[:, None]
    d = x - co
    n = np.linalg.norm(d, axis=1)
    d *= np.minimum(1.0, RELIEF_CLAMP_M / np.maximum(n, 1e-12))[:, None]
    return d


def bake_native(obj):
    """store the native MPFB relief (default macros, no project deformation, no point offsets) on the mesh"""
    st = C.get_state(obj)
    hook, C.POINT_HOOK = getattr(C, "POINT_HOOK", None), None
    ground = obj.get("cs_ground_z")
    try:
        if ground is not None:
            del obj["cs_ground_z"]
        C.apply_state(obj, C.default_state())
        co = C.raw_coords(obj)
        ctx = SC._ctx(obj)
        r = relief(co, ctx)
        H = float(np.ptp(co[ctx["body"], 2]))
    finally:
        C.POINT_HOOK = hook
        if ground is not None:
            obj["cs_ground_z"] = ground
        C.apply_state(obj, st)
    a = obj.data.attributes.get(NAT_ATTR) or obj.data.attributes.new(NAT_ATTR, 'FLOAT', 'POINT')
    a.data.foreach_set("value", r.astype(np.float32))
    obj["cs_native_H"] = H
    return r, H


_DONE = []


def register():
    if _DONE:
        return
    _ORIG_APPLY.append(SC.apply_custom)
    SC.apply_custom = hook                 # s1_custom.register() (called again by s1_panel.boot) installs SC.apply_custom = this wrapper
    SC.register()
    for c in V3_CONTROLS:
        if c["id"] not in SC.CUSTOM_IDS:
            SC.CUSTOM.append(c)
            SC.CUSTOM_IDS.append(c["id"])
        if c["id"] not in C.CTRL:
            C.CONTROLS.append(c)
            C.CTRL[c["id"]] = c
    SC._DEFORMERS["v3_bustc_back"] = d_v3_bustc_back
    SC._DEFORMERS["v3_back_narrow"] = d_v3_back_narrow
    SC._DEFORMERS["v3_relief"] = d_v3_relief
    SC._DEFORMERS["v3_fatdist"] = d_v3_fatdist
    w = C.CTRL["weight"]
    w["lo"], w["hi"] = 0.0, 1.0            # v3: MPFB's own full weight range (fat_to_state maps 胖瘦 -> weight)
    p = list(SC._POST)
    if "v3_relief" not in p:
        p.insert(p.index("back_relax") + 1, "v3_relief")
    SC._POST = tuple(p)
    _DONE.append(1)
