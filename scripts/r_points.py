"""v019 RED points (红点 = bust / glute SHAPE design; third tier after green = proportion and white = posture / local).

Rules:
  * a red point is an ordinary entry of the point registry (a_points) with  red=True, stage="B".  It owns ONLY red sliders
    (s1_custom.RED_IDS) and has NO local offset (local_axes = ""): dragging it can never change a green / white slider, so the
    proportions and the posture set by the upper tiers stay as they are.
  * every red point sits where its own slider moves the surface MOST (measured on the current surface: the slider is moved
    -0.5 / +0.5 and the left-side vertices with the largest response along the dragged axis are taken) - so the point is on the
    place it shapes and 1 unit of the slider moves it as much as possible.  Stored as vertex ids, they follow every later edit.
  * the red sliders are symmetric (one value for both sides): a red point always acts on both sides.
"""
import json

import numpy as np

import s1_core as C
import s1_custom as SC
import a_points as A

RED_GROUPS = ["红·胸", "红·臀"]
WHITE_HIDE = ("w_bust_", "w_glute_")        # their white points are hidden by default once the red points exist (not deleted)


def _r(id, label, group, side, hint, owners, region, parent, main=True, radius=6):
    axes = "".join(a for a in "XYZ" if a in owners)
    sp = A._spec(id, label, group, side, hint, main=main, axes=axes, local_axes="", owners=owners, region=region, radius=radius,
                 parent=parent, source="红点（v019）：放在它的红滑块作用最大的表面位置（本机测量选取）")
    sp.update(red=True, stage="B")
    return sp


# point -> (driving slider used to place it, axis index of the response)
PLACE = {"r_bust_upper": ("r_bust_upper", 1), "r_bust_low": ("r_bust_lower", 1), "r_imf": ("r_imf", 2),
         "r_cleavage": ("r_cleavage", 0), "r_side_bust": ("r_side_bust", 0),
         "r_glute_shelf": ("r_glute_shelf", 1), "r_glute_lowfull": ("r_glute_lowfull", 1), "r_glute_fold": ("r_glute_fold_d", 1),
         "r_glute_hollow": ("r_glute_hollow", 0)}


def definitions():
    D = []
    for s, w in (("l", "左"), ("r", "右")):
        S = s.upper()
        bust, glute = ["bust", "ribcage"], ["glute", "hip", "thigh"]
        D += [
            _r("r_bust_upper_" + s, w + "上胸坡", "红·胸", S, "前后拖：上胸坡凹进（水滴形）↔ 饱满（圆形）；乳头不动", {"Y": ["r_bust_upper"]}, bust, "bust_root_" + s),
            _r("r_nipple_" + s, w + "乳头朝向", "红·胸", S, "上下拖：乳头朝上 ↔ 朝下；左右拖：朝外 ↔ 朝内；只动胸尖，胸根不动",
               {"Z": ["r_nip_pitch"], "X": ["r_nip_yaw"]}, bust, "bust_root_" + s),
            _r("r_bust_low_" + s, w + "下胸弧", "红·胸", S, "前后拖：下胸弧更圆更兜 ↔ 更扁", {"Y": ["r_bust_lower"]}, bust, "bust_root_" + s),
            _r("r_imf_" + s, w + "乳下线", "红·胸", S, "上下拖：乳房下缘那条线提高 ↔ 降低（作用小，最多约 3 mm）", {"Z": ["r_imf"]}, bust, "bust_root_" + s),
            _r("r_cleavage_" + s, w + "乳沟侧", "红·胸", S, "左右拖：乳沟变宽 ↔ 更聚拢；胸骨中线不动", {"X": ["r_cleavage"]}, bust, "bust_root_" + s),
            _r("r_side_bust_" + s, w + "侧胸", "红·胸", S, "左右拖：侧胸（靠腋下）丰满 ↔ 收进", {"X": ["r_side_bust"]}, bust, "bust_root_" + s),
            _r("r_glute_shelf_" + s, w + "上臀", "红·臀", S, "前后拖：上臀（臀峰以上）翘起饱满 ↔ 收平", {"Y": ["r_glute_shelf"]}, glute, "glute_center_" + s),
            _r("r_glute_lowfull_" + s, w + "下臀", "红·臀", S, "前后拖：下臀（臀下线以上）饱满 ↔ 瘪", {"Y": ["r_glute_lowfull"]}, glute, "glute_center_" + s),
            _r("r_glute_fold_" + s, w + "臀下线", "红·臀", S, "前后拖：臀下线压深 ↔ 变浅；上下拖：臀下线提高 ↔ 降低",
               {"Y": ["r_glute_fold_d"], "Z": ["r_glute_fold_h"]}, glute, "glute_center_" + s),
            _r("r_glute_hollow_" + s, w + "侧臀凹", "红·臀", S, "左右拖：侧臀（大转子后方）凹进 ↔ 填平；胯宽测量处不动", {"X": ["r_glute_hollow"]}, glute,
               "hip_outer_" + s),
        ]
    return D


def _response(obj, cid, axis, X0):
    """per-vertex response of slider `cid` along `axis` (m per unit, central difference around the current value)"""
    st0 = C.get_state(obj)
    v0 = float(st0.get(cid, 0.0))
    a, b = max(v0 - 0.5, -1.0), min(v0 + 0.5, 1.0)
    st = dict(st0)
    st[cid] = a
    C.apply_state(obj, st)
    Xa = C.raw_coords(obj).copy()
    st[cid] = b
    C.apply_state(obj, st)
    Xb = C.raw_coords(obj).copy()
    C.apply_state(obj, st0)
    return (Xb[:, axis] - Xa[:, axis]) / (b - a)


def _anchors(obj):
    co = C.raw_coords(obj).copy()
    zone = C.load_zones(obj)
    left = (co[:, 0] > 0.004) & (zone != 0)
    S = {}
    for pid, (cid, ax) in PLACE.items():
        r = np.abs(_response(obj, cid, ax, co))
        r[~left] = 0.0
        S[pid + "_l"] = np.argsort(-r)[:3]
    nip = C.group_indices(obj, "nipple")
    S["r_nipple_l"] = nip[co[nip, 0] > 0]
    return S


def build(obj):
    rc = A.rest_co(obj)
    mv = np.array(list(obj["cs_mvmap"]), dtype=np.int32)
    S = _anchors(obj)
    specs = []
    for sp in definitions():
        pid = sp["id"]
        ids = S[pid] if pid in S else mv[S[pid[:-2] + "_l"]]
        sp["verts"] = [int(i) for i in np.unique(ids)]
        sp["rest"] = [round(float(v), 5) for v in rc[sp["verts"]].mean(0)]
        sp["mirror"] = pid[:-2] + ("_r" if pid.endswith("_l") else "_l")
        specs.append(sp)
    return specs


def extend_object(obj):
    """v019 upgrade: append the red points (anchors measured on the CURRENT surface, red sliders at their current values) and hide the
    white bust / glute points by default (main = False; still listed and draggable in their groups).  Idempotent.
    Returns (added ids, white points hidden)."""
    reg = json.loads(obj["cs_points"])
    have = {p["id"] for p in reg}
    new = [sp for sp in build(obj) if sp["id"] not in have]
    hidden = 0
    for p in reg:
        if p.get("white") and p["id"].startswith(WHITE_HIDE) and p["main"]:
            p["main"] = False
            hidden += 1
    obj["cs_points"] = json.dumps(reg + new)
    A.load(obj)
    return [sp["id"] for sp in new], hidden


def is_red(p):
    return bool(p.get("red"))


def sensitivity(obj):
    """-> {point id: {slider: mm per unit along its axis}} measured at the current state (validation / report)"""
    out = {}
    X0 = C.raw_coords(obj).copy()
    for sp in definitions():
        if not sp["id"].endswith("_l"):
            continue
        p = A.P.by_id.get(sp["id"])
        if p is None:
            continue
        row = {}
        for key, ids in p["owners"].items():
            ax = "XYZ".index(key)
            for cid in ids:
                r = _response(obj, cid, ax, X0)
                row[cid] = round(float(np.mean(r[p["verts"]])) * 1000, 2)
        out[sp["id"]] = row
    return out
