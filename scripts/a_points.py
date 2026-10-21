"""Stage A green-point system (控制点：定位 + 拖动 + 镜像 + 保护约束).  Runs inside Blender, no GUI needed.

Design (task book v2 §A3 / §8):
  * Every point is a stable id with an ANCHOR = a fixed set of vertex indices chosen once on the neutral body
    (joint cubes of the MPFB rig groups, or surface vertices picked by geometry).  Its position is always evaluated from
    the CURRENT mesh, so every slider / preset / NL edit moves the points with the body (nothing is pinned in world space).
  * Dragging a point changes the MODEL in two ways, both stored in the character state:
      "ctrl"  : mirror ON  -> solve the owned MPFB / trial sliders (finite-difference Jacobian + bounded least squares).
                Sliders are left/right symmetric by construction, so the opposite side follows exactly.
      "local" : the residual (what the sliders cannot reach, or everything when mirror is OFF) goes into a per-point
                OFFSET.  An offset is a smooth displacement field: weights come from mesh-edge hop distance (independent of
                the pose, so a re-opened file reproduces it bit-for-bit), limited to a body region, plus an optional
                rigid "follow" region (children of the chain move with the parent, e.g. knee -> shin and foot).
  * Mirror: on = L/R share one value; off = independent offsets.  Turning it on again with a real asymmetry present
    asks for an explicit source (left / right / average) - nothing is overwritten silently.
  * Midline points keep x = 0 (axes "YZ").  Each point lists the axes it may move on; ground contact and left/right
    order are protection constraints checked by a_health.
  * Face placeholders (eyes / ears / mouth) only RECORD a position for stage C (record_only) - they do not deform the
    mesh in stage A.  This is stated in the point hint and in the test table.
State that must survive a save: object custom properties cs_points (registry), cs_pt_off (offsets), cs_pt_mirror.
"""
import json
import math
import types

import numpy as np
import bpy

import s1_core as C

MAX_OFF = 0.08                  # m, hard cap of one point offset
# measured with a_health on v007 (docs/S1_test_log.md): beyond these local offsets the mesh health check starts to report new spikes/creases/folds
MAX_OFF_BY = {"bust_root_": 0.01, "waist_side_": 0.03, "hip_outer_": 0.03, "glute_center_": 0.02, "pelvis_side_": 0.03,
              # stage-B white points: overall ceiling only; the real per-direction limits are measured (assets/white_point_caps.json, b_white_scan.py)
              "w_": 0.035}


def _load_white_caps():
    import os
    try:
        d = json.load(open(os.path.join(C.ROOT, "assets", "white_point_caps.json"), encoding="utf-8"))
        return d["caps"]
    except (OSError, KeyError, ValueError):
        return {}


WHITE_CAPS = _load_white_caps()      # {left/mid point id: {"x-": m, "x+": m, ...}}; missing = no measurement -> conservative 4 mm


def dir_caps(pid):
    """per-direction local-offset limits (m) of a white point, or None for green points. Right-side points use the mirrored left entry."""
    if not pid.startswith("w_"):
        return None
    if pid.endswith("_r"):
        c = WHITE_CAPS.get(pid[:-2] + "_l")
        return None if c is None else {k[0] + ("-" if k[1] == "+" else "+") if k[0] == "x" else k: v for k, v in c.items()}
    return WHITE_CAPS.get(pid)


def clamp_dir(pid, o):
    """clamp an offset vector component-wise by the measured per-direction caps; returns (vector, clamped?)"""
    caps = dir_caps(pid)
    if caps is None:
        return o, False
    o = np.array(o, dtype=float)
    hit = False
    for i, a in enumerate("xyz"):
        if abs(o[i]) < 1e-9:
            continue
        lim = caps.get(a + ("+" if o[i] > 0 else "-"), 0.004 if not caps else None)
        if lim is None:
            continue                                    # axis not tested (locked axis)
        if abs(o[i]) > lim:
            o[i] = math.copysign(lim, o[i])
            hit = True
    return o, hit


def max_off(pid):
    for k, v in MAX_OFF_BY.items():
        if pid.startswith(k):
            return v
    return MAX_OFF


MIN_SENS_RED = 0.002            # m per unit for RED points: their sliders are bounded (-1..1) and are the ONLY thing a red point may change
                                # (no local offset), so a weak owner (r_imf: ~3 mm per unit) is still used - the bounded solve just stops at 1
MIN_SENS = 0.004                # m per control unit: owners that move the point less than this are not inverted (found with the bust root: dragging it forward SHRANK the bust)
RESID_TOL = 0.0015              # m, residual below which no local offset is added

P = types.SimpleNamespace(reg=[], by_id={}, off={}, mirror=True, wcache={}, mvmap=None, adj=None, obj_name=None,
                          groups_open=set(), show=True, selected=None, health_red=set(), vis_extra=set(),
                          show_green=True, show_white=True, show_red=True, show_shape=True)
# show_red = the mesh-health markers (drawn PURPLE since v019); show_shape = the v019 red shape points (红点 = bust / glute shape design)

GROUPS = ["中轴", "肩臂", "下肢", "骨盆与轮廓", "胸部", "臀部", "头脸占位", "手足占位",
          "白·肩腋", "白·手臂", "白·胸", "白·背", "白·腰腹", "白·臀", "白·腿", "白·手掌", "白·手指(默认隐藏)", "白·脚", "白·脚趾(默认隐藏)",
          "红·胸", "红·臀"]
GROUP_MAIN_OPEN = {"中轴", "骨盆与轮廓", "胸部", "肩臂", "下肢", "臀部", "红·胸", "红·臀"}      # groups whose MAIN points are drawn by default


# ------------------------------------------------------------------ registry
def _spec(id, label, group, side, hint, verts=None, vg=None, vg_half=None, main=False, parent=None, axes="XYZ",
          local_axes=None, owners=None, region=(), follow=(), follow_dir=None, radius=14, kind="surface", record_only=False,
          source="", range_info=None, keep=()):
    """owners: {"X": [ctrl ids], "Y": [...], "Z": [...]} = sliders solved per axis ("*" = limb-length sliders solved
    together on all axes).  What the sliders cannot deliver goes into the point's local offset."""
    owners = owners or {}
    return dict(id=id, label=label, group=group, side=side, hint=hint, verts=verts, vg=vg, vg_half=vg_half, main=main,
                parent=parent, axes=axes, local_axes=axes if local_axes is None else local_axes, owners={k: list(v) for k, v in owners.items()},
                region=list(region), follow=list(follow), follow_dir=follow_dir, radius=radius, kind=kind,
                record_only=record_only, source=source, range=range_info, keep=list(keep))


def _pair(spec_fn):
    """spec_fn(side_letter, side_word, other_id_suffix) -> spec for 'l' and 'r'."""
    return [spec_fn("l", "左"), spec_fn("r", "右")]


def definitions():
    """Static part of the registry (everything except vertex indices, which depend on the mesh)."""
    D = []
    # ---- 中轴 (midline, x locked)
    D += [
        _spec("head_center", "头部中心", "中轴", "M", "定位头颈位置。上下拖动=颈长（头跟着走）；前后=只改头部局部。阶段A不验收脸。", vg=["joint-l-eye", "joint-r-eye", "joint-head"],
              main=True, axes="YZ", owners={"Z": ["neck_len"]}, region=["head", "neck"], follow=["head"], radius=10, kind="joint",
              source="MPFB 关节顶点组（双眼+颅底中心）"),
        _spec("neck_front", "颈根前参考", "中轴", "M", "颈根前表面位置参考（锁骨窝上方）；拖动改颈前后厚度（局部）", main=False, axes="YZ",
              region=["neck"], radius=7, parent="head_center", source="颈部表面顶点（几何选取）"),
        _spec("neck_back", "颈后参考", "中轴", "M", "颈后表面（第七颈椎附近）；与头/胸廓/肩共同约束姿态，保留背部曲线；拖动改颈后位置（局部）",
              main=False, axes="YZ", region=["neck", "ribcage"], radius=9, parent="head_center", source="颈部表面顶点（几何选取）"),
        _spec("bust_ctrl", "胸部中央总控", "中轴", "M", "两侧乳尖连线中点的工程控制点（不是解剖点）。上下=胸位置，前后=胸前凸/圆润；左右对称整体联动",
              vg=["nipple"], main=True, axes="YZ", owners={"Z": ["bust_pos"], "Y": ["bust_round", "cupsize"]}, region=["bust", "ribcage"], radius=16, kind="control",
              parent="ribcage_center", source="乳尖顶点组中点"),
        _spec("ribcage_center", "胸廓中心", "中轴", "M", "上身（胸廓）位置；上下=胸廓在腰与颈之间上移/下移（身高不变），前后=胸廓前后", vg=["joint-spine-2"], main=True, axes="YZ",
              owners={"Y": ["torso_fb"]}, region=["ribcage", "bust", "waist", "neck"], radius=22, kind="joint", parent="pelvis_center",
              source="MPFB 关节顶点组 spine-2"),
        _spec("pelvis_center", "骨盆中心", "中轴", "M", "骨盆整体位置；上下=腿长↔躯干长分配（身高不变），前后=骨盆前后位移（站姿重心）", vg=["joint-pelvis"], main=True, axes="YZ",
              owners={"Z": ["thigh_len", "torso_len"], "Y": ["hip_fb"]}, region=["hip", "glute", "waist"], radius=18, kind="joint", source="MPFB 关节顶点组 pelvis",
              keep=["H"]),
        _spec("perineum", "会阴区域总控", "中轴", "M", "只表示该区域位置/形状占位，阶段A不做细节；拖动改下腹/耻骨前凸或局部位置", main=False, axes="YZ",
              owners={"Y": ["pelvis_bulge"]}, region=["hip", "thigh"], radius=6, parent="pelvis_center", source="胯底中线顶点（几何选取）"),
    ]
    # ---- 肩臂
    for s, w in (("l", "左"), ("r", "右")):
        D += [
            _spec("shoulder_surf_" + s, w + "肩表面参考", "肩臂", s.upper(), "肩部表面（三角肌上外缘）参考，与肩关节中心分开记录；左右=肩宽，上下=肩高低",
                  main=True, axes="XYZ", owners={"X": ["shoulder_w"]}, region=["uarm", "ribcage", "neck"], radius=11, parent="ribcage_center",
                  source="肩部表面顶点（几何选取）"),
            _spec("shoulder_joint_" + s, w + "肩关节中心", "肩臂", s.upper(), "肩关节中心（MPFB 骨架位置，不等于肩峰）；拖动改肩宽/整条手臂位置",
                  vg=["joint-%s-shoulder" % s], main=False, axes="XYZ", owners={"X": ["shoulder_w"]}, region=["uarm", "ribcage"], follow=["uarm", "larm", "hand"],
                  follow_dir="down", radius=12, kind="joint", parent="ribcage_center", source="MPFB 关节顶点组"),
            _spec("elbow_" + s, w + "肘", "肩臂", s.upper(), "上臂/前臂分界；沿臂方向拖动=上臂长度", vg=["joint-%s-elbow" % s], axes="XYZ", owners={"*": ["uarm_len"]},
                  region=["uarm", "larm"], follow=["larm", "hand"], follow_dir="down", radius=10, kind="joint", parent="shoulder_joint_" + s,
                  source="MPFB 关节顶点组"),
            _spec("wrist_" + s, w + "腕", "肩臂", s.upper(), "前臂/手分界；沿臂方向拖动=前臂长度", vg=["joint-%s-hand" % s], axes="XYZ", owners={"*": ["larm_len"]},
                  region=["larm", "hand"], follow=["hand"], follow_dir="down", radius=8, kind="joint", parent="elbow_" + s, source="MPFB 关节顶点组"),
        ]
    # ---- 下肢
    for s, w in (("l", "左"), ("r", "右")):
        D += [
            _spec("hip_joint_" + s, w + "髋关节参考", "下肢", s.upper(), "髋关节中心（MPFB 关节，与胯外侧宽度不是同一点）；左右=腿间距/胯宽，上下=腿长↔躯干长分配（身高不变）",
                  vg=["joint-%s-upper-leg" % s], main=True, axes="XYZ", local_axes="XY", owners={"X": ["hip_w"], "Z": ["thigh_len", "torso_len"]}, keep=["H"],
                  region=["thigh", "calf", "foot", "hip", "glute"], follow=["thigh", "calf", "foot"], follow_dir="down", radius=14, kind="joint",
                  parent="pelvis_center", source="MPFB 关节顶点组"),
            _spec("knee_" + s, w + "膝", "下肢", s.upper(), "膝关节；上下=膝的高低（大腿↔小腿分配，腿长和身高不变），左右=膝间距（并拢），前后=膝线", vg=["joint-%s-knee" % s], main=True, axes="XYZ",
                  local_axes="XY", owners={"Z": ["shin_len", "thigh_len"], "X": ["knees_in"], "Y": ["knees_back"]}, keep=["H"], region=["thigh", "calf", "foot"], follow=["calf", "foot"], follow_dir="down",
                  radius=14, kind="joint", parent="hip_joint_" + s, source="MPFB 关节顶点组"),
            _spec("ankle_" + s, w + "踝", "下肢", s.upper(), "踝关节；左右前后=脚位置（并腿/脚间距）。上下锁定=地面接触保护，腿长请拖髋/膝", vg=["joint-%s-ankle" % s], main=True, axes="XY",
                  owners={"X": ["legs_together"]}, region=["calf", "foot", "thigh"], follow=["foot"], follow_dir="down",
                  radius=10, kind="joint", parent="knee_" + s, source="MPFB 关节顶点组"),
        ]
    # ---- 骨盆与轮廓
    for s, w in (("l", "左"), ("r", "右")):
        D += [
            _spec("waist_side_" + s, w + "腰侧参考", "骨盆与轮廓", s.upper(), "腰最细处的侧轮廓（整体控制）；左右=腰围，前后=腰厚", main=True, axes="XY", owners={"X": ["waist_c"]},
                  region=["waist", "ribcage", "hip"], radius=12, parent="pelvis_center", source="腰侧表面顶点（几何选取，非精确骨点）"),
            _spec("pelvis_side_" + s, w + "骨盆侧参考", "骨盆与轮廓", s.upper(), "骨盆侧缘（髂嵴高度）参考，不假称从图精确测得骨点；左右=骨盆宽",
                  main=False, axes="XY", owners={"X": ["hip_w"]}, region=["hip", "waist"], radius=12, parent="pelvis_center", source="骨盆侧表面顶点（几何选取）"),
            _spec("hip_outer_" + s, w + "胯外侧参考", "骨盆与轮廓", s.upper(), "胯最宽处的外侧轮廓；左右=胯宽（含软组织），与髋关节分开", main=True, axes="XY",
                  owners={"X": ["hip_c"]}, region=["hip", "glute", "thigh"], radius=14, parent="pelvis_center", source="胯外侧表面顶点（几何选取）"),
        ]
    # ---- 胸部
    for s, w in (("l", "左"), ("r", "右")):
        D += [
            _spec("bust_root_" + s, w + "乳根中心", "胸部", s.upper(), "胸与胸廓连接区域的中心（不是锁骨连接点）；并记录该区域的宽/高/朝向。上下=胸位置，左右=间距，前后=大小/前凸",
                  main=True, axes="XYZ", owners={"Z": ["bust_pos"], "X": ["bust_dist"], "Y": ["cupsize"]}, region=["bust", "ribcage"], radius=14, parent="ribcage_center",
                  source="胸区域边界环中心（几何选取）", range_info=True),
            _spec("nipple_" + s, w + "乳尖位置参考", "胸部", s.upper(), "乳尖/乳头位置参考（阶段A不做乳头外观）；拖动改胸的朝向与高度间距", vg=["nipple"], vg_half=s, main=False,
                  axes="XYZ", owners={"Z": ["bust_pos"], "X": ["bust_dist"], "Y": ["bust_round"]}, region=["bust"], radius=9, parent="bust_root_" + s, kind="joint",
                  source="MPFB 乳头顶点组"),
        ]
    # ---- 臀部
    for s, w in (("l", "左"), ("r", "右")):
        D += [
            _spec("glute_center_" + s, w + "臀部中心", "臀部", s.upper(), "臀部后突最大处；前后=后突大形，左右=臀宽，上下=臀峰高度", main=True, axes="XYZ",
                  owners={"Y": ["glute_proj", "glute_vol"]}, region=["glute", "hip", "thigh"], radius=13, parent="pelvis_center",
                  source="臀区最凸顶点（几何选取）"),
        ]
    # ---- 头脸占位（只记录位置）
    for s, w in (("l", "左"), ("r", "右")):
        D += [
            _spec("eye_" + s, w + "眼中心", "头脸占位", s.upper(), "占位：只记录位置供阶段C使用，阶段A不改网格也不验收五官", vg=["joint-%s-eye" % s], axes="XYZ",
                  region=["head"], kind="joint", record_only=True, parent="head_center", source="MPFB 关节顶点组"),
            _spec("ear_" + s, w + "耳参考", "头脸占位", s.upper(), "占位：只记录位置供阶段C使用", vg=["ears"], vg_half=s, axes="XYZ", region=["head"], record_only=True,
                  parent="head_center", source="MPFB 耳朵顶点组"),
        ]
    D += [_spec("mouth_center", "嘴中心", "头脸占位", "M", "占位：只记录位置供阶段C使用", vg=["joint-mouth"], axes="YZ", region=["head"], kind="joint",
                record_only=True, parent="head_center", source="MPFB 关节顶点组")]
    # ---- 手足占位（局部形变，定方向长度）
    for s, w in (("l", "左"), ("r", "右")):
        D += [
            _spec("palm_" + s, w + "掌心", "手足占位", s.upper(), "手方向与位置占位（整只手跟随）；阶段A不验收手指细节", vg=["joint-%s-finger-3-1" % s], axes="XYZ",
                  region=["hand", "larm"], follow=["hand"], follow_dir="all", radius=8, kind="joint", parent="wrist_" + s, source="MPFB 关节顶点组"),
            _spec("middle_tip_" + s, w + "中指尖参考", "手足占位", s.upper(), "手长度/指向占位", vg=["joint-%s-finger-3-4" % s], axes="XYZ", region=["hand"], radius=6,
                  kind="joint", parent="palm_" + s, source="MPFB 关节顶点组"),
            _spec("heel_" + s, w + "足跟", "手足占位", s.upper(), "足占地范围后端（地面接触受保护，不可上下拖）", axes="XY", region=["foot"], radius=8, parent="ankle_" + s,
                  source="足底后端顶点（几何选取）"),
            _spec("toe_in_" + s, w + "大脚趾/前足内侧", "手足占位", s.upper(), "足前端内侧（足长与朝向占位；不验收脚趾细节）", vg=["joint-%s-toe-1-3" % s], axes="XY", region=["foot"],
                  radius=7, kind="joint", parent="ankle_" + s, source="MPFB 关节顶点组"),
            _spec("toe_out_" + s, w + "前足外侧", "手足占位", s.upper(), "足前端外侧（足宽与朝向占位）", vg=["joint-%s-toe-5-3" % s], axes="XY", region=["foot"], radius=7, kind="joint",
                  parent="ankle_" + s, source="MPFB 关节顶点组"),
        ]
    return D


def _mirror_map(obj, co):
    """vertex -> mirrored vertex (x -> -x) from the NEUTRAL, left/right symmetric body.  Stored on the object."""
    from mathutils import kdtree
    n = len(co)
    kd = kdtree.KDTree(n)
    for i in range(n):
        kd.insert(co[i], i)
    kd.balance()
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        _, j, d = kd.find((-co[i, 0], co[i, 1], co[i, 2]))
        out[i] = j
    return out


def _adjacency(obj):
    ed = C._adjacency(obj)
    return ed[:, 0], ed[:, 1]


def build_registry(obj):
    """Choose anchor vertices on the NEUTRAL body (default state, no offsets), store registry + mirror map on the object.
    The current state is restored afterwards."""
    st0 = C.get_state(obj)
    off0, mir0 = P.off, P.mirror
    P.off = {}
    for pr in ("cs_rest_x", "cs_rest_y", "cs_rest_z"):
        if pr in obj:
            del obj[pr]
    _REST.clear()
    C.apply_state(obj, C.default_state())
    try:
        reg = _build_registry(obj)
    finally:
        P.off, P.mirror = off0, mir0
        C.apply_state(obj, st0)
    return reg


def _build_registry(obj):
    co = C.raw_coords(obj)
    for a in range(3):
        _rest(obj, a)
    zone = C.load_zones(obj)
    ZM = lambda *keys: np.isin(zone, [C.ZID[k] for k in keys])
    keep = zone != 0
    zmin, zmax = float(co[keep, 2].min()), float(co[keep, 2].max())
    H = zmax - zmin
    L = C.landmarks(obj, co)
    ea, eb = _adjacency(obj)
    mvmap = _mirror_map(obj, co)
    left = co[:, 0] > 0.004

    def vg(name, half=None):
        ids = C.group_indices(obj, name)
        if half == "l":
            ids = ids[co[ids, 0] > 0]
        elif half == "r":
            ids = ids[co[ids, 0] < 0]
        return ids

    def at_height(mask, fl, tol, axis, sign, k=4):
        z0 = zmin + fl * H
        idx = np.where(mask & (np.abs(co[:, 2] - z0) < tol))[0]
        order = np.argsort(-sign * co[idx, axis])[:k]
        return idx[order]

    sel = {}
    shoulder = co[vg("joint-l-shoulder")].mean(0)
    cand = np.where(ZM("uarm", "ribcage") & left & (np.linalg.norm(co - shoulder, axis=1) < 0.075))[0]
    score = 0.7 * (co[cand, 0] - shoulder[0]) + 0.7 * (co[cand, 2] - shoulder[2])
    sel["shoulder_surf_l"] = cand[np.argsort(-score)[:4]]
    sel["hip_outer_l"] = at_height(ZM("hip", "glute", "waist", "thigh") & left, 0.525, 0.012, 0, +1)
    sel["pelvis_side_l"] = at_height(ZM("hip", "waist") & left, 0.575, 0.012, 0, +1)
    sel["waist_side_l"] = at_height(ZM("waist", "ribcage") & left, 0.65, 0.012, 0, +1)
    g = np.where(ZM("glute") & left)[0]
    peak = g[np.argmax(co[g, 1])]
    sel["glute_center_l"] = g[np.linalg.norm(co[g] - co[peak], axis=1) < 0.03]
    # bust root = ring of the bust zone on the chest wall (zone boundary vertices of the +x bust)
    bz = ZM("bust") & left
    ring = np.zeros(len(co), bool)
    m1 = bz[ea] & ~bz[eb] & keep[eb]
    ring[ea[m1]] = True
    m2 = bz[eb] & ~bz[ea] & keep[ea]
    ring[eb[m2]] = True
    sel["bust_root_l"] = np.where(ring & bz)[0]
    nz = L["neck_z"] - 0.012
    mid = ZM("neck") & (np.abs(co[:, 0]) < 0.02) & (np.abs(co[:, 2] - nz) < 0.012)
    idx = np.where(mid)[0]
    sel["neck_front"] = idx[np.argsort(co[idx, 1])[:3]]
    sel["neck_back"] = idx[np.argsort(-co[idx, 1])[:3]]
    body = C.group_indices(obj, "body")
    cm = np.zeros(len(co), bool)
    cm[body] = True
    cm &= keep & (np.abs(co[:, 0]) < 0.012) & (co[:, 2] > zmin + 0.35 * H) & (co[:, 2] < zmin + 0.55 * H)
    idx = np.where(cm)[0]
    sel["perineum"] = idx[np.argsort(co[idx, 2])[:4]]
    f = np.where(ZM("foot") & left & (co[:, 2] < zmin + 0.035))[0]
    sel["heel_l"] = f[np.argsort(-co[f, 1])[:4]]

    reg = definitions()
    for sp in reg:
        pid = sp["id"]
        ids = None
        if sp["vg"]:
            parts = [vg(n, sp["vg_half"]) for n in sp["vg"]]
            ids = np.concatenate(parts) if parts else None
        elif pid in sel:
            ids = sel[pid]
        elif pid.endswith("_r") and pid[:-2] + "_l" in sel:
            ids = mvmap[sel[pid[:-2] + "_l"]]
        if ids is None or len(ids) == 0:
            raise RuntimeError("no anchor vertices for point " + pid)
        sp["verts"] = [int(i) for i in np.unique(ids)]
        sp["rest"] = [round(float(v), 5) for v in co[sp["verts"]].mean(0)]
        sp["mirror"] = (pid[:-2] + ("_r" if pid.endswith("_l") else "_l")) if sp["side"] in ("L", "R") else None
    obj["cs_ground_z"] = float(co[C.group_indices(obj, "body"), 2].min())   # ground-contact protection (see s1_custom.apply_custom)
    obj["cs_points"] = json.dumps(reg)
    obj["cs_mvmap"] = [int(i) for i in mvmap]
    obj["cs_pt_mirror"] = 1
    obj["cs_pt_off"] = "{}"
    load(obj)
    return reg


def load(obj):
    """Read registry / offsets / mirror from the object (call after opening a file)."""
    P.obj_name = obj.name
    raw = obj.get("cs_points")
    P.reg = json.loads(raw) if raw else []
    P.by_id = {p["id"]: p for p in P.reg}
    P.off = {k: list(map(float, v)) for k, v in json.loads(obj.get("cs_pt_off", "{}")).items()}
    P.mirror = bool(obj.get("cs_pt_mirror", 1))
    mv = obj.get("cs_mvmap")
    P.mvmap = np.array(list(mv), dtype=np.int32) if mv is not None else None
    P.wcache = {}
    P.adj = None
    P.selected = None
    P.health_red = set()
    C.POINT_HOOK = offsets_displacement
    C.HELPER_FOLLOW = _nearest_body
    return P.reg


GREEN_V019 = ("ribcage_center", "pelvis_center", "hip_joint_l", "hip_joint_r", "knee_l", "knee_r")


def upgrade_green_v019(obj):
    """v019: copy the new owners / keep / hint / local axes of the joint points (keep the stature while dragging Z; ribcage Z = local)
    from definitions() into the registry stored on the object.  Anchors / offsets untouched.  Returns the ids changed."""
    reg = json.loads(obj["cs_points"])
    D = {d["id"]: d for d in definitions()}
    changed = []
    for p in reg:
        if p["id"] in GREEN_V019:
            d = D[p["id"]]
            new = dict(owners=d["owners"], keep=d["keep"], hint=d["hint"], local_axes=d["local_axes"])
            if any(p.get(k) != v for k, v in new.items()):
                p.update(new)
                changed.append(p["id"])
    obj["cs_points"] = json.dumps(reg)
    load(obj)
    return changed


def save_props(obj):
    obj["cs_pt_off"] = json.dumps({k: [round(x, 6) for x in v] for k, v in P.off.items() if any(abs(x) > 1e-9 for x in v)})
    obj["cs_pt_mirror"] = 1 if P.mirror else 0


# ------------------------------------------------------------------ evaluation
def positions(obj, co=None):
    co = C.raw_coords(obj) if co is None else co
    return {p["id"]: co[p["verts"]].mean(axis=0) for p in P.reg}


def pos(obj, pid, co=None):
    co = C.raw_coords(obj) if co is None else co
    return co[P.by_id[pid]["verts"]].mean(axis=0)


def mirror_vec(v):
    return [-v[0], v[1], v[2]]


# ------------------------------------------------------------------ weight fields (hop distance, pose independent)
def _csr(obj):
    if P.adj is None:
        ea, eb = _adjacency(obj)
        P.adj = (ea, eb)
    return P.adj


def _hops(obj, seeds, region, rmax):
    ea, eb = _csr(obj)
    n = len(obj.data.vertices)
    dist = np.full(n, 1e9)
    fr = np.zeros(n, bool)
    fr[seeds] = True
    dist[seeds] = 0
    for h in range(1, int(rmax) + 1):
        a, b = fr[ea], fr[eb]
        nxt = np.zeros(n, bool)
        nxt[eb[a]] = True
        nxt[ea[b]] = True
        nxt &= (dist > 1e8) & region
        if not nxt.any():
            break
        dist[nxt] = h
        fr = nxt
    return dist


def _smooth(f, ea, eb, iters):
    f = f.astype(np.float64).copy()
    cnt = np.ones(len(f))
    np.add.at(cnt, ea, 1)
    np.add.at(cnt, eb, 1)
    for _ in range(iters):
        acc = f.copy()
        np.add.at(acc, ea, f[eb])
        np.add.at(acc, eb, f[ea])
        f = acc / cnt
    return f


def rest_co(obj):
    """Neutral-body coordinates (stored once on the object at registry build)."""
    return np.stack([_rest(obj, a) for a in range(3)], axis=1)


def rest_x(obj):
    return _rest(obj, 0)


def rest_z(obj):
    return _rest(obj, 2)


_REST = {}


def _rest(obj, axis):
    key = (obj.name, len(obj.data.vertices), axis)
    if key in _REST:
        return _REST[key]
    prop = "cs_rest_%s" % "xyz"[axis]
    if obj.get(prop) is not None:
        arr = np.array(list(obj[prop]), dtype=np.float64)
    else:
        arr = C.raw_coords(obj)[:, axis]
        obj[prop] = [float(v) for v in arr]            # stored once: the neutral-body coordinate (build time)
    _REST[key] = arr
    return arr


def _body_kd(obj):
    key = ("kd", obj.name, len(obj.data.vertices))
    if key in P.wcache:
        return P.wcache[key]
    from mathutils import kdtree
    zone = C.load_zones(obj)
    rc = rest_co(obj)
    body = np.where(zone != 0)[0]
    kd = kdtree.KDTree(len(body))
    for j, i in enumerate(body):
        kd.insert(rc[i], j)
    kd.balance()
    P.wcache[key] = (kd, body)
    return kd, body


def _nearest_body(obj):
    """For every vertex: nearest non-helper vertex on the NEUTRAL body (helper joint cubes follow the surface around them)."""
    key = ("nb", obj.name, len(obj.data.vertices))
    if key in P.wcache:
        return P.wcache[key]
    zone = C.load_zones(obj)
    rc = rest_co(obj)
    kd, body = _body_kd(obj)
    nb = np.arange(len(zone))
    for i in np.where(zone == 0)[0]:
        nb[i] = body[kd.find(rc[i])[1]]
    P.wcache[key] = nb
    return nb


def _seed_body(obj, sp):
    """Seed vertices on the body surface: the anchors themselves, or (joint cubes in the helper zone) the 8 body vertices
    nearest to the anchor centre."""
    zone = C.load_zones(obj)
    v = np.array(sp["verts"], dtype=int)
    if (zone[v] != 0).any():
        return v[zone[v] != 0]
    rc = rest_co(obj)
    kd, body = _body_kd(obj)
    c = rc[v].mean(axis=0)
    return np.array([body[j] for _, j, _ in kd.find_n(c, 8)], dtype=int)


def weights(obj, pid):
    """Per-vertex weight of the offset of point `pid` (cached; depends on topology, zones and the neutral body only, so a
    re-opened file reproduces every offset exactly)."""
    key = (pid, len(obj.data.vertices))
    w = P.wcache.get(key)
    if w is not None:
        return w
    sp = P.by_id[pid]
    zone = C.load_zones(obj)
    ea, eb = _csr(obj)
    nb = _nearest_body(obj)
    side = sp["side"]
    xr = rest_x(obj)
    zr = rest_z(obj)

    def side_mask(m):
        if side == "L":
            return m & (xr > -0.004)
        if side == "R":
            return m & (xr < 0.004)
        if side == "M":
            return m & (np.abs(xr) < 0.16)
        return m
    region = side_mask(np.isin(zone, [C.ZID[k] for k in sp["region"]]))
    seeds = _seed_body(obj, sp)
    region[seeds] = True
    dist = _hops(obj, seeds, region, sp["radius"])
    t = np.clip(1.0 - dist / float(sp["radius"]), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)                                 # smoothstep falloff
    if sp["follow"]:
        fz = side_mask(np.isin(zone, [C.ZID[k] for k in sp["follow"]]))
        if sp["follow_dir"] == "down":
            fz &= zr < float(np.mean(zr[seeds])) + 0.01
        w = np.where(fz, 1.0, w)
    # soften the region / follow boundary so that no crease is born at its edge
    w = np.clip(_smooth(w, ea, eb, 4), 0.0, 1.0)
    allowed = np.isin(zone, [C.ZID[k] for k in set(sp["region"]) | set(sp["follow"])])
    w[~allowed] = 0.0
    helper = zone == 0
    w[helper] = w[nb[helper]]                                  # helper vertices (joint cubes) follow the body around them
    w[np.array(sp["verts"], dtype=int)] = 1.0                  # the point's own anchors move exactly with its offset
    w = w.astype(np.float32)
    P.wcache[key] = w
    return w



def offsets_displacement(obj):
    """Hook called by s1_custom.apply_custom: sum of all point offsets as a per-vertex displacement (or None)."""
    if not P.reg or not any(any(abs(x) > 1e-9 for x in v) for v in P.off.values()):
        return None
    d = np.zeros((len(obj.data.vertices), 3))
    for pid, v in P.off.items():
        if pid not in P.by_id or P.by_id[pid]["record_only"]:
            continue
        v = np.array(v)
        if np.abs(v).max() < 1e-9:
            continue
        d += weights(obj, pid)[:, None].astype(np.float64) * v[None, :]
    return d


# ------------------------------------------------------------------ dragging
EPS_BI, EPS_MACRO = 0.12, 0.05
KEEP_W = 10.0                   # weight of the "keep" rows (e.g. body height) against the point-position rows in the least squares


def body_height(obj, co=None):
    """top of the body (helper cubes excluded) - the ground is held fixed by apply_custom, so this is the stature"""
    co = C.raw_coords(obj) if co is None else co
    return float(co[C.load_zones(obj) != 0, 2].max())


def _axes_mask(sp, which="axes"):
    s = sp[which]
    return np.array([1.0 if a in s else 0.0 for a in "XYZ"])


def snapshot():
    return dict(off={k: list(v) for k, v in P.off.items()}, mirror=P.mirror)


def restore(obj, snap):
    P.off = {k: list(v) for k, v in snap["off"].items()}
    P.mirror = bool(snap["mirror"])
    save_props(obj)


def _bounded_ls(J, delta, cur, lo, hi, lam=1e-4):
    n = J.shape[1]
    lam = lam * max(float((J * J).sum(axis=0).max()), 1e-12)          # relative damping
    dc = np.zeros(n)
    free = np.ones(n, bool)
    for _ in range(n + 1):
        A = J[:, free]
        if A.shape[1] == 0:
            break
        rhs = delta - J[:, ~free] @ dc[~free]
        sol = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ rhs)
        dc[free] = sol
        new = cur + dc
        over = (new > hi + 1e-12) | (new < lo - 1e-12)
        if not (over & free).any():
            break
        for i in np.where(over & free)[0]:
            dc[i] = (hi[i] if new[i] > hi[i] else lo[i]) - cur[i]
            free[i] = False
    return dc


class Drag:
    """One drag gesture = Drag(obj, pid) ... update(target) x N ... (the caller records ONE history entry at the end).
    update() is idempotent w.r.t. the state at construction, so jittery mouse paths never accumulate."""

    def __init__(self, obj, pid):
        self.obj, self.pid = obj, pid
        self.sp = P.by_id[pid]
        self.st0 = C.get_state(obj)
        self.off0 = {k: list(v) for k, v in P.off.items()}
        self.X0 = C.raw_coords(obj).copy()
        self.p0 = pos(obj, pid, self.X0)
        self.dX = {}              # full-mesh response of every owner slider (per unit) - the cheap live preview superposes these
        self.kb0 = None
        self.h0 = None            # mesh-health baseline at gesture start (white points only: owners + local offset can add up to new spikes/creases)
        if pid.startswith(("w_", "r_")):           # v019: red points too (several red sliders together can tip a fold crease)
            import a_health as H
            self.h0 = H.check(obj)
        self.mir_on = P.mirror and self.sp["side"] in ("L", "R", "M")
        self.groups = []          # (axis indices, [ctrl ids], J sub-matrix, keep rows or None)
        self.weak = []            # (axis key, control label) owners dropped for too low sensitivity
        if not self.sp["record_only"] and self.mir_on and self.sp["owners"]:
            self._jacobian()

    def _jacobian(self):
        obj, sp = self.obj, self.sp
        allowed = [i for i, a in enumerate("XYZ") if a in sp["axes"]]
        cols = {}
        keep = list(sp.get("keep") or [])
        kcols = {}
        h0 = body_height(obj) if keep else 0.0
        self.h_keep = h0
        for key, ids in sp["owners"].items():
            axes = allowed if key == "*" else [i for i, a in enumerate("XYZ") if a == key and a in sp["axes"]]
            for cid in ids:
                if cid not in C.CTRL or not axes:
                    continue
                c = C.CTRL[cid]
                eps = 0.25 if c["kind"] == "custom" else (EPS_MACRO if c["kind"] == "macro" else EPS_BI)
                v0 = self.st0[cid]
                v1 = v0 + eps if v0 + eps <= c["hi"] + 1e-9 else v0 - eps
                st = dict(self.st0)
                st[cid] = v1
                C.apply_state(obj, st)
                X = C.raw_coords(obj)
                col = (pos(obj, self.pid, X) - self.p0) / (v1 - v0)
                self.dX[cid] = ((X - self.X0) / (v1 - v0)).astype(np.float32)
                kc = np.array([(body_height(obj) - h0) / (v1 - v0)]) if keep and 2 in axes else None
                if np.linalg.norm(col[axes]) < (MIN_SENS_RED if sp.get("red") else MIN_SENS) and (kc is None or abs(kc[0]) < MIN_SENS):
                    self.weak.append((key, C.CTRL[cid]["label"]))      # ill-conditioned: a 1-unit change moves this point < 4 mm, so inverting it gives huge / wrong-signed steps
                    continue
                cols[(key, cid)] = col
                if kc is not None:
                    kcols[(key, cid)] = kc
        C.apply_state(obj, self.st0)
        for key in sp["owners"]:
            ids = [cid for (k, cid) in cols if k == key]
            if not ids:
                continue
            axes = allowed if key == "*" else [i for i, a in enumerate("XYZ") if a == key]
            J = np.stack([cols[(key, cid)][axes] for cid in ids], axis=1)
            K = np.stack([kcols[(key, cid)] for cid in ids], axis=1) if all((key, cid) in kcols for cid in ids) else None
            self.groups.append((axes, ids, J, K))

    def update(self, target, guard=True):
        """Move the point towards `target` (world position).  Returns (state, messages).
        White points are health-guarded: if the combined slider + local-offset result adds a NEW spike/crease/fold, the whole
        change is scaled back (bisection) to the largest fraction that adds none."""
        st, msg = self._update(target)
        if not guard or self.h0 is None or self.sp["record_only"]:       # guard=False: live preview while the mouse moves (the guard runs on release)
            return st, msg
        import a_health as H
        if not H.new_since(self.h0, H.check(self.obj)):
            return st, msg
        st1, off1 = dict(st), {k: list(v) for k, v in P.off.items()}
        lo, hi, best = 0.0, 1.0, 0.0
        for _ in range(6):
            t = 0.5 * (lo + hi)
            self._apply_fraction(st1, off1, t)
            if H.new_since(self.h0, H.check(self.obj)):
                hi = t
            else:
                lo = best = t
        self._apply_fraction(st1, off1, best)
        msg.append("【再往这个方向拖会新增尖刺/折痕，已自动收到安全位置（约 %d%% 的拖动量）；更大的移动请换滑块或超出支持范围】" % round(best * 100))
        return {k: float(v) for k, v in C.get_state(self.obj).items()}, msg

    def _offset_for(self, o_new):
        """cap / clamp a proposed local offset exactly like _update does -> (offset, messages)"""
        sp, msg = self.sp, []
        o = np.array(o_new, dtype=float)
        n = np.linalg.norm(o)
        cap = max_off(self.pid)
        if n > cap:
            o = o * cap / n
            msg.append("【局部偏移已到上限 %.0f cm】" % (cap * 100))
        o, hit = clamp_dir(self.pid, o)
        if hit:
            msg.append("【该方向局部偏移已到实测上限】")
        if sp["side"] == "M":
            o[0] = 0.0
        return o, msg

    def preview(self, target):
        """FAST live preview while the mouse moves (~10 ms instead of ~180 ms): no MPFB / custom-deformer recompute - the owner
        sliders' effect is the linear superposition of the full-mesh responses measured at gesture start, plus the local offset
        weights.  It writes only the display shape key (cs_custom); the exact result (update(), health guard) is computed on
        release, and cancel() restores everything.  Returns messages."""
        obj, sp = self.obj, self.sp
        keys = obj.data.shape_keys
        kb = keys.key_blocks.get("cs_custom") if keys is not None else None
        target = np.array(target, dtype=float)
        if kb is None or sp["record_only"] or not np.isfinite(target).all():
            self.update(target, guard=False)
            return []
        if self.kb0 is None:
            self.kb0 = np.empty(len(kb.data) * 3, dtype=np.float32)
            kb.data.foreach_get("co", self.kb0)
            self.kb0 = self.kb0.reshape(-1, 3)
        delta = (target - self.p0) * _axes_mask(sp)
        if sp["side"] == "M":
            delta[0] = 0.0
        X = self.X0.astype(np.float32, copy=True)
        moved = np.zeros(3)
        msg = []
        for axes, ids, J, K in self.groups:
            d = delta[axes]
            if np.abs(d).max() < 1e-5:
                continue
            cur = np.array([self.st0[c] for c in ids])
            lo = np.array([C.CTRL[c]["lo"] for c in ids])
            hi = np.array([C.CTRL[c]["hi"] for c in ids])
            if K is not None:
                dc = _bounded_ls(np.vstack([J, KEEP_W * K]), np.concatenate([d, np.zeros(len(K))]), cur, lo, hi)
            else:
                dc = _bounded_ls(J, d, cur, lo, hi)
            moved[axes] += J @ dc
            for c, x in zip(ids, dc):
                if abs(x) > 1e-6:
                    X += np.float32(x) * self.dX[c]
                    msg.append("%s %+.2f" % (C.CTRL[c]["label"], x))
        resid = (target - (self.p0 + moved)) * _axes_mask(sp, "local_axes")
        if sp["side"] == "M":
            resid[0] = 0.0
        o_old = np.array(self.off0.get(self.pid, [0.0, 0.0, 0.0]))
        if np.linalg.norm(resid) > RESID_TOL:
            o, m2 = self._offset_for(o_old + resid)
            msg += m2
            X += weights(obj, self.pid)[:, None] * (o - o_old).astype(np.float32)[None, :]
            if self.mir_on and sp["mirror"]:
                om = np.array(self.off0.get(sp["mirror"], [0.0, 0.0, 0.0]))
                X += weights(obj, sp["mirror"])[:, None] * (np.array(mirror_vec(o)) - om).astype(np.float32)[None, :]
        kb.data.foreach_set("co", (self.kb0 + (X - self.X0.astype(np.float32))).reshape(-1))
        obj.data.update()
        C._RAW_LAST[obj.name] = X.astype(np.float64)
        return ["预览中（松开鼠标后精确计算 + 网格健康检查）"] + msg

    def _apply_fraction(self, st1, off1, t):
        st = {k: self.st0[k] + t * (st1[k] - self.st0[k]) for k in self.st0}
        keys = set(self.off0) | set(off1)
        P.off = {k: [a + t * (b - a) for a, b in zip(self.off0.get(k, [0.0] * 3), off1.get(k, [0.0] * 3))] for k in keys}
        save_props(self.obj)
        C.apply_state(self.obj, st)

    def _update(self, target):
        obj, sp = self.obj, self.sp
        target = np.array(target, dtype=float)
        if not np.isfinite(target).all():
            C.apply_state(obj, self.st0)
            return dict(self.st0), ["非有限数值（NaN/inf）：已拒绝，模型保持拖动前的状态"]
        delta = (target - self.p0) * _axes_mask(sp)
        if sp["side"] == "M":
            delta[0] = 0.0
        st = dict(self.st0)
        off = {k: list(v) for k, v in self.off0.items()}
        msg = []
        if sp["record_only"]:
            P.off = off
            P.off[self.pid] = [float(x) for x in (np.array(off.get(self.pid, [0, 0, 0])) + delta)]
            if self.mir_on and sp["mirror"]:
                P.off[sp["mirror"]] = mirror_vec(P.off[self.pid])
            save_props(obj)
            C.apply_state(obj, st)
            return st, ["占位点：只记录位置（阶段C使用），模型未改动"]
        if sp.get("red") and not self.groups:
            C.apply_state(obj, st)
            return st, ["红点只做左右对称的造型（红滑块两侧共用）：请先打开「左右镜像」再拖"]
        for axes, ids, J, K in self.groups:
            d = delta[axes]
            if np.abs(d).max() < 1e-5:
                continue
            cur = np.array([st[c] for c in ids])
            lo = np.array([C.CTRL[c]["lo"] for c in ids])
            hi = np.array([C.CTRL[c]["hi"] for c in ids])
            if K is not None:                                   # e.g. keep the stature: the point moves, the total height does not
                dc = _bounded_ls(np.vstack([J, KEEP_W * K]), np.concatenate([d, np.zeros(len(K))]), cur, lo, hi)
            else:
                dc = _bounded_ls(J, d, cur, lo, hi)
            for c, x, a, b in zip(ids, dc, lo, hi):
                st[c] = float(st[c] + x)
                if abs(x) > 1e-4:
                    msg.append("%s %+.2f" % (C.CTRL[c]["label"], x))
                if (x > 0 and abs(st[c] - b) < 1e-6) or (x < 0 and abs(st[c] - a) < 1e-6):
                    msg.append("【%s 已到范围%s】" % (C.CTRL[c]["label"], "上限" if x > 0 else "下限"))
        if any(K is not None for _, _, _, K in self.groups):
            self._keep_refine(st, delta)
        for key, lab in self.weak:
            ax = [i for i, a in enumerate("XYZ") if (a == key or key == "*")]
            if any(abs(delta[i]) > 1e-3 for i in ax):
                msg.append("【%s 对该点几乎没有影响，不用它；这个方向只做局部偏移（有上限）】" % lab)
        locked = (target - self.p0) * (1.0 - _axes_mask(sp))
        if np.abs(locked).max() > 0.002:
            msg.append("【%s 方向已锁定（%s）】" % ("".join(a for i, a in enumerate("XYZ") if locked[i] and abs(locked[i]) > 0.002),
                                                "中线/地面接触/占位点保护"))
        P.off = off
        C.apply_state(obj, st)
        now = pos(obj, self.pid)
        resid = (target - now) * _axes_mask(sp, "local_axes")
        if sp["side"] == "M":
            resid[0] = 0.0
        if np.linalg.norm(resid) > RESID_TOL:
            o = np.array(P.off.get(self.pid, [0.0, 0.0, 0.0])) + resid
            n = np.linalg.norm(o)
            cap = max_off(self.pid)
            if n > cap:
                o = o * cap / n
                msg.append("【局部偏移已到上限 %.0f cm（再大网格会出尖刺/折痕），更大的移动超出支持范围】" % (cap * 100))
            o, hit = clamp_dir(self.pid, o)
            if hit:
                msg.append("【该方向局部偏移已到实测上限（再大网格会新增尖刺/折痕）；更大的移动请用滑块或超出支持范围】")
            if sp["side"] == "M":
                o[0] = 0.0
            P.off[self.pid] = [float(x) for x in o]
            if self.mir_on and sp["mirror"]:
                P.off[sp["mirror"]] = mirror_vec(P.off[self.pid])
            save_props(obj)
            C.apply_state(obj, st)
            now = pos(obj, self.pid)
            msg.append("局部偏移 %.1f cm" % (np.linalg.norm(o) * 100))
        else:
            save_props(obj)
        miss = (target - now) * _axes_mask(sp)
        if np.linalg.norm(miss) > 0.004:
            msg.append("未完全到位，差 %.1f cm（该方向受保护/超出范围）" % (np.linalg.norm(miss) * 100))
        return st, msg

    def _keep_refine(self, st, delta, steps=3):
        """MPFB length sliders are two targets (incr / decr), so their slope changes at 0 and the start Jacobian can be wrong for
        the direction actually taken: re-measure and correct with a Broyden-updated Jacobian so the kept quantity (stature)
        really stays put (measured: 2.7 mm drift -> < 1 mm)."""
        obj = self.obj
        for axes, ids, J, K in self.groups:
            if K is None:
                continue
            M = np.vstack([J, K]).astype(float)
            w = np.concatenate([np.ones(len(axes)), KEEP_W * np.ones(len(K))])
            want = np.concatenate([delta[axes], np.zeros(len(K))])
            lo = np.array([C.CTRL[c]["lo"] for c in ids])
            hi = np.array([C.CTRL[c]["hi"] for c in ids])
            x0 = np.array([self.st0[c] for c in ids])
            y0 = np.zeros(len(want))
            x_prev, y_prev = x0, y0
            x = np.array([st[c] for c in ids])
            for _ in range(steps):
                for c, v in zip(ids, x):
                    st[c] = float(v)
                C.apply_state(obj, st)
                y = np.concatenate([(pos(obj, self.pid) - self.p0)[axes], [body_height(obj) - self.h_keep] * len(K)])
                if abs(y[-1]) < 3e-4 and np.abs(y[:-len(K)] - want[:-len(K)]).max() < 5e-4:
                    break
                dx, dy = x - x_prev, y - y_prev
                if float(dx @ dx) > 1e-12:
                    M = M + np.outer(dy - M @ dx, dx) / float(dx @ dx)
                x_prev, y_prev = x, y
                dc = _bounded_ls(M * w[:, None], (want - y) * w, x, lo, hi)
                x = x + dc
            for c, v in zip(ids, x):
                st[c] = float(v)

    def cancel(self):
        P.off = {k: list(v) for k, v in self.off0.items()}
        save_props(self.obj)
        C.apply_state(self.obj, self.st0)



def drag_point(obj, pid, delta_world):
    """Convenience for tests / NL: one complete gesture moving the point by `delta_world`."""
    d = Drag(obj, pid)
    st, msg = d.update(d.p0 + np.array(delta_world, dtype=float))
    return d, st, msg


# ------------------------------------------------------------------ mirror state
def asymmetry(obj):
    """Largest |L - mirror(R)| over offset vectors (m)."""
    m = 0.0
    for p in P.reg:
        if p["side"] == "L" and p["mirror"]:
            a = np.array(P.off.get(p["id"], [0, 0, 0]))
            b = np.array(mirror_vec(P.off.get(p["mirror"], [0, 0, 0])))
            m = max(m, float(np.abs(a - b).max()))
    return m


def set_mirror(obj, on, source=None):
    """on=True with an existing asymmetry needs source in ('L','R','AVG'); returns (ok, message)."""
    if not on:
        P.mirror = False
        save_props(obj)
        return True, "镜像已关闭：现在拖动只改被拖的一侧（局部偏移）；中线点仍锁定在对称面"
    asym = asymmetry(obj)
    if asym > 1e-4 and source is None:
        return False, "两侧现在不对称（最大差 %.1f cm）。重新开启镜像前请选择以哪一侧为准：左 / 右 / 取平均" % (asym * 100)
    if asym > 1e-4:
        for p in P.reg:
            if p["side"] != "L" or not p["mirror"]:
                continue
            a = np.array(P.off.get(p["id"], [0, 0, 0]))
            b = np.array(mirror_vec(P.off.get(p["mirror"], [0, 0, 0])))
            v = a if source == "L" else (b if source == "R" else 0.5 * (a + b))
            P.off[p["id"]] = [float(x) for x in v]
            P.off[p["mirror"]] = mirror_vec(P.off[p["id"]])
    P.mirror = True
    save_props(obj)
    return True, "镜像已开启" + ("（两侧按「%s」统一）" % {"L": "以左为准", "R": "以右为准", "AVG": "取平均"}[source] if asym > 1e-4 else "")


def reset_points(obj, ids=None):
    """Clear local offsets of `ids` (all when None) and put their owned sliders back to the draft values."""
    ids = list(P.by_id) if ids is None else list(ids)
    draft = None
    if obj.get("cs_draft_state"):
        draft = json.loads(obj["cs_draft_state"])
    st = C.get_state(obj)
    owned = set()
    for pid in ids:
        sp = P.by_id[pid]
        P.off.pop(pid, None)
        if sp["mirror"] and sp["mirror"] not in ids and P.mirror:
            P.off.pop(sp["mirror"], None)
        for cids in sp["owners"].values():          # (v019 fix: was owned.update(owners) = the AXIS keys, so owned sliders were never reset)
            owned.update(cids)
    for cid in owned:
        if cid not in C.CTRL:
            continue
        if draft and cid in draft:
            st[cid] = draft[cid]
        elif C.CTRL[cid].get("stage") == "B":         # stage-B sliders are not in the stage-A draft: back to neutral
            st[cid] = C.CTRL[cid]["default"]
    save_props(obj)
    C.apply_state(obj, st)
    return st


def export_points(obj, extra=None):
    """Registry + live positions + offsets as plain JSON (task book §8 item 2)."""
    co = C.raw_coords(obj)
    out = dict(mirror=P.mirror, groups=GROUPS, points=[])
    for p in P.reg:
        v = co[p["verts"]].mean(axis=0)
        out["points"].append(dict(id=p["id"], label=p["label"], group=p["group"], side=p["side"], stage=p.get("stage", "A"), white=bool(p.get("white")), red=bool(p.get("red")), parent=p["parent"],
                                  mirror_id=p["mirror"], axes=p["axes"], main=p["main"], kind=p["kind"], source=p["source"],
                                  owners={k: v for k, v in p["owners"].items()}, record_only=p["record_only"], hint=p["hint"], n_anchor_verts=len(p["verts"]),
                                  position=[round(float(x), 5) for x in v], rest=p["rest"],
                                  offset=[round(x, 5) for x in P.off.get(p["id"], [0, 0, 0])], image_observation=None,
                                  influence_region=p["region"], follow_region=p["follow"], radius_hops=p["radius"]))
    if extra:
        out.update(extra)
    return out


# ------------------------------------------------------------------ visibility
_INT = {}


def interior_ids(obj):
    """points that sit inside the body (joint / centre points anchored on helper geometry, > 10 mm below the skin at the rest pose).
    The viewport draws and picks them 'x-ray' (a depth test would hide them behind the skin, so they could never be grabbed)."""
    key = (obj.name, len(obj.data.vertices))
    if key in _INT:
        return _INT[key]
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    co = rest_co(obj)
    zone = C.load_zones(obj)
    polys = [tuple(pg.vertices) for pg in obj.data.polygons if all(zone[i] != 0 for i in pg.vertices)]
    bvh = BVHTree.FromPolygons([tuple(v) for v in co], polys)
    out = set()
    for p in P.reg:
        q = Vector(co[p["verts"]].mean(axis=0))
        loc, nrm, idx, dist = bvh.find_nearest(q)
        if dist is not None and dist > 0.010:
            out.add(p["id"])
    _INT[key] = out
    return out


def point_visible(obj, pid, w, toward, tol=0.025):
    """visible from a view whose vector (object -> camera) is `toward`: interior points always; surface points when no skin lies more than
    `tol` in front of them, or when their skin faces the camera / is at the silhouette (normal . toward > -0.05)"""
    from mathutils import Vector
    if pid in interior_ids(obj):
        return True
    toward = Vector(toward).normalized()
    ok, loc, nrm, idx = obj.ray_cast(Vector(w) + toward * 0.4, -toward)
    if not ok or (loc - Vector(w)).dot(toward) < tol:
        return True
    ok2, loc2, nrm2, idx2 = obj.closest_point_on_mesh(Vector(w))
    return bool(ok2 and (loc2 - Vector(w)).length < 0.012 and nrm2.dot(toward) > -0.05 and (loc - Vector(w)).dot(toward) < 0.06)


def is_visible(p):
    if not P.show:
        return False
    if p.get("red"):
        if not P.show_shape:
            return False
    elif p.get("white"):
        if not P.show_white:
            return False
    elif not P.show_green:
        return False
    return p["main"] or p["group"] in P.groups_open or p["id"] in P.vis_extra


def bust_range(obj, co=None):
    """Range width / height / direction of each bust root (task book A3 补充规则), from the ring of the bust zone."""
    co = C.raw_coords(obj) if co is None else co
    out = {}
    for s in ("l", "r"):
        sp = P.by_id.get("bust_root_" + s)
        if sp is None:
            continue
        R = co[sp["verts"]]
        nip = co[P.by_id["nipple_" + s]["verts"]].mean(0)
        c = R.mean(0)
        d = nip - c
        out[s] = dict(center=[float(x) for x in c], width=float(R[:, 0].max() - R[:, 0].min()), height=float(R[:, 2].max() - R[:, 2].min()),
                      direction=[float(x) for x in d / (np.linalg.norm(d) + 1e-9)])
    return out
