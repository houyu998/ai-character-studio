"""Stage B white points (白点 = local surface-feature controls; the green points of stage A stay the global-shape controls).

Rules (task book v2 §4 step 4):
  * a white point is an ordinary entry of the point registry (a_points) with  white=True, stage="B": stored in the object's
    `cs_points`, dragged by a_points.Drag (owned sliders first, the rest becomes a capped LOCAL offset), mirrored through
    `cs_mvmap`, saved in `cs_pt_off`, undo/redo through the normal history.
  * defaults: every finger / toe point is HIDDEN (their groups are closed and none is `main`); every point mirrors.
  * a slider is owned only by a point whose trait cannot be reached by dragging alone (girth, fullness, orientation...).
  * the influence of a point is computed by a_points.weights (hop distance inside the listed body regions, smoothstep) - never hand painted.
Anchors are picked once on the NEUTRAL body (cs_rest_* coordinates) and stored as vertex ids, so they follow every later edit.
"""
import json

import numpy as np

import s1_core as C
import a_points as A

WHITE_GROUPS = ["白·肩腋", "白·手臂", "白·胸", "白·背", "白·腰腹", "白·臀", "白·腿", "白·手掌", "白·手指(默认隐藏)", "白·脚", "白·脚趾(默认隐藏)"]
HIDDEN_GROUPS = ("白·手指(默认隐藏)", "白·脚趾(默认隐藏)")


def _w(id, label, group, side, hint, owners=None, main=False, axes="XYZ", region=(), radius=7, parent=None, vg=None, kind="surface",
       record_only=False):
    sp = A._spec(id, label, group, side, hint, vg=vg, main=main, axes=axes, owners=owners, region=region, radius=radius, parent=parent,
                 kind=kind, record_only=record_only,
                 source="白点（阶段B）：" + ("MPFB 关节顶点组" if vg else "中性体型表面几何选取"))
    sp.update(white=True, stage="B")
    return sp


def definitions():
    D = []
    for s, w in (("l", "左"), ("r", "右")):
        S = s.upper()
        D += [
            _w("w_deltoid_" + s, w + "三角肌外缘", "白·肩腋", S, "肩外侧轮廓；局部改肩头圆润度", main=True, region=["uarm", "ribcage"], radius=6,
               parent="shoulder_surf_" + s, owners={"X": ["shoulder_muscle"]}),
            _w("w_armpit_f_" + s, w + "腋前", "白·肩腋", S, "腋窝前缘（胸与上臂交界）", region=["uarm", "ribcage", "bust"], radius=5, parent="shoulder_surf_" + s),
            _w("w_armpit_b_" + s, w + "腋后", "白·肩腋", S, "腋窝后缘（背阔肌/大圆肌交界）", region=["uarm", "ribcage"], radius=5, parent="shoulder_surf_" + s),
            _w("w_armpit_" + s, w + "腋窝底", "白·肩腋", S, "腋窝最低处；拖得太用力会出折角（健康检查会标红）", main=True, region=["uarm", "ribcage"], radius=5,
               parent="shoulder_surf_" + s),
            _w("w_acromion_" + s, w + "肩峰", "白·肩腋", S, "肩峰（肩最高处）；局部改肩头高低/圆润（v018 新增）", main=True, region=["uarm", "ribcage"], radius=5,
               parent="shoulder_surf_" + s),
            _w("w_trap_" + s, w + "斜方肌上缘（肩颈斜坡）", "白·肩腋", S, "肩颈斜坡；局部改肩颈厚度（v018 新增）", region=["ribcage", "uarm"], radius=6, parent="shoulder_surf_" + s),
            _w("w_scap_" + s, w + "肩胛（上背外侧）", "白·背", S, "上背外侧（肩胛处）；往前拖=这一块背变薄（v018 新增）", main=True, region=["ribcage", "uarm"], radius=7,
               parent="shoulder_surf_" + s),
            _w("w_back_up_" + s, w + "上背中部（脊柱旁）", "白·背", S, "脊柱旁的上背；往前拖=背变薄（v018 新增）", main=True, region=["ribcage"], radius=7, parent="shoulder_surf_" + s),
            _w("w_uarm_out_" + s, w + "上臂外缘", "白·手臂", S, "上臂外侧轮廓；左右=上臂围", main=True, owners={"X": ["uarm_c"]}, region=["uarm"], radius=7, parent="elbow_" + s),
            _w("w_uarm_in_" + s, w + "上臂内缘", "白·手臂", S, "上臂内侧轮廓（局部）", region=["uarm"], radius=7, parent="elbow_" + s),
            _w("w_larm_out_" + s, w + "前臂外缘", "白·手臂", S, "前臂外侧轮廓；左右=前臂脂肪", main=True, owners={"X": ["larm_fat"]}, region=["larm"], radius=7, parent="wrist_" + s),
            _w("w_larm_in_" + s, w + "前臂内缘", "白·手臂", S, "前臂内侧轮廓", region=["larm"], radius=7, parent="wrist_" + s),
            _w("w_bust_in_" + s, w + "胸内缘（乳沟侧）", "白·胸", S, "胸内侧轮廓/乳沟边；左右=胸间距", main=True, owners={"X": ["bust_dist"]}, region=["bust", "ribcage"], radius=6,
               parent="bust_root_" + s),
            _w("w_bust_out_" + s, w + "胸外缘", "白·胸", S, "胸外侧轮廓；左右=外展/内收朝向", main=True, owners={"X": ["bust_yaw"]}, region=["bust", "ribcage"], radius=6,
               parent="bust_root_" + s),
            _w("w_bust_up_" + s, w + "胸上缘", "白·胸", S, "胸上轮廓/锁骨下过渡；前后=上半球饱满", main=True, owners={"Y": ["bust_upper"]}, region=["bust", "ribcage"], radius=6,
               parent="bust_root_" + s),
            _w("w_bust_low_" + s, w + "胸下缘", "白·胸", S, "胸下轮廓；前后=下半球饱满，上下=下垂", main=True, owners={"Y": ["bust_lowfull"], "Z": ["bust_sag"]},
               region=["bust", "ribcage"], radius=6, parent="bust_root_" + s),
            _w("w_glute_top_" + s, w + "臀上缘", "白·臀", S, "臀上沿（腰臀过渡）", region=["glute", "hip", "waist"], radius=6, parent="glute_center_" + s),
            _w("w_glute_peak_" + s, w + "臀峰", "白·臀", S, "臀最凸处（比绿点“臀部中心”更局部）；上下=臀峰高度", main=True, owners={"Z": ["glute_up"]}, region=["glute", "hip"],
               radius=6, parent="glute_center_" + s),
            _w("w_glute_out_" + s, w + "臀外缘", "白·臀", S, "臀外侧轮廓（含软组织）", main=True, region=["glute", "hip", "thigh"], radius=6, parent="hip_outer_" + s),
            _w("w_glute_low_" + s, w + "臀下缘", "白·臀", S, "臀下沿（臀腿分界）", main=True, region=["glute", "thigh"], radius=6, parent="glute_center_" + s),
            _w("w_glute_leg_" + s, w + "臀腿过渡", "白·臀", S, "臀下缘到大腿后侧的过渡；整体柔化请用“臀↔腿过渡柔化”", region=["glute", "thigh"], radius=6,
               parent="glute_center_" + s),
            _w("w_thigh_out_" + s, w + "大腿外缘", "白·腿", S, "大腿外侧轮廓；左右=大腿围", main=True, owners={"X": ["thigh_c"]}, region=["thigh", "hip"], radius=8, parent="knee_" + s),
            _w("w_thigh_in_" + s, w + "大腿内缘", "白·腿", S, "大腿内侧轮廓", region=["thigh"], radius=8, parent="knee_" + s),
            _w("w_knee_f_" + s, w + "膝前", "白·腿", S, "膝盖前面；前后=膝围（局部）", owners={"X": ["knee_c"]}, region=["thigh", "calf"], radius=5, parent="knee_" + s),
            _w("w_calf_out_" + s, w + "小腿外缘", "白·腿", S, "小腿外侧轮廓；左右=小腿围", main=True, owners={"X": ["calf_c"]}, region=["calf"], radius=7, parent="knee_" + s),
            _w("w_calf_in_" + s, w + "小腿内缘", "白·腿", S, "小腿内侧轮廓", region=["calf"], radius=7, parent="knee_" + s),
            _w("w_calf_in_up_" + s, w + "小腿内缘上段", "白·腿", S, "小腿内侧上段轮廓（膝下，v018 新增）", main=True, region=["calf"], radius=6, parent="knee_" + s),
            _w("w_calf_in_low_" + s, w + "小腿内缘下段", "白·腿", S, "小腿内侧下段轮廓（踝上，v018 新增）", main=True, region=["calf"], radius=6, parent="knee_" + s),
            _w("w_shin_" + s, w + "小腿前缘（胫骨）", "白·腿", S, "小腿前面（胫骨线，v018 新增）", region=["calf"], radius=6, parent="knee_" + s),
            _w("w_calf_b_" + s, w + "小腿肚", "白·腿", S, "小腿肚最凸处；前后=小腿肌肉", main=True, owners={"Y": ["calf_muscle"]}, region=["calf"], radius=7, parent="knee_" + s),
            _w("w_palm_out_" + s, w + "手掌外缘（小指侧）", "白·手掌", S, "手掌小指侧边缘", region=["hand"], radius=4, parent="palm_" + s),
            _w("w_palm_in_" + s, w + "手掌内缘（虎口）", "白·手掌", S, "手掌拇指侧/虎口", region=["hand"], radius=4, parent="palm_" + s),
            _w("w_foot_in_" + s, w + "足内缘（足弓侧）", "白·脚", S, "脚内侧边缘（足弓上方）", region=["foot"], radius=5, axes="XY", parent="ankle_" + s),
            _w("w_foot_out_" + s, w + "足外缘", "白·脚", S, "脚外侧边缘", region=["foot"], radius=5, axes="XY", parent="ankle_" + s),
            _w("w_inst_" + s, w + "足背最高处", "白·脚", S, "脚背最高点（踝前）；上下锁定（地面保护），只能前后左右", region=["foot", "calf"], radius=5, axes="XY", parent="ankle_" + s),
            _w("w_foot_fore_" + s, w + "前足最宽处", "白·脚", S, "前脚掌最宽处；左右=前足宽", owners={"X": ["foot_w"]}, region=["foot"], radius=5, axes="XY", parent="ankle_" + s),
        ]
        for f, fname in ((1, "拇指"), (2, "食指"), (3, "中指"), (4, "无名指"), (5, "小指")):
            for j, jn in ((2, "指关节"), (4, "指尖")):
                D.append(_w("w_finger_%d_%d_%s" % (f, j, s), "%s%s%s" % (w, fname, jn), "白·手指(默认隐藏)", S,
                            "手指局部（默认隐藏）" + ("；沿指方向=手指长度" if j == 4 else ""), vg=["joint-%s-finger-%d-%d" % (s, f, j)],
                            owners={"*": ["finger_len"]} if j == 4 else None, region=["hand"], radius=3, parent="palm_" + s, kind="joint"))
        for t, tname in ((1, "大脚趾"), (2, "二趾"), (3, "三趾"), (4, "四趾"), (5, "小趾")):
            tip = 3 if t == 1 else 4
            for j, jn in ((1, "趾根"), (tip, "趾尖")):
                D.append(_w("w_toe_%d_%d_%s" % (t, j, s), "%s%s%s" % (w, tname, jn), "白·脚趾(默认隐藏)", S, "脚趾局部（默认隐藏；上下锁定=地面保护）",
                            vg=["joint-%s-toe-%d-%d" % (s, t, j)], region=["foot"], radius=3, axes="XY", parent="ankle_" + s, kind="joint"))
    D += [
        _w("w_waist_front", "腰前中线", "白·腰腹", "M", "腰前中线表面；前后=腹部紧实/柔软（与肚脐很近，默认不显示）", axes="YZ", main=False, region=["waist", "hip"], radius=7, parent="pelvis_center",
           owners={"Y": ["abd_tone"]}),
        _w("w_navel", "肚脐", "白·腰腹", "M", "肚脐位置；上下=肚脐高低，前后=凹凸", axes="YZ", main=True, region=["waist"], radius=4, parent="pelvis_center",
           owners={"Z": ["navel_ud"], "Y": ["navel_io"]}),
        _w("w_lower_abd", "下腹中线", "白·腰腹", "M", "肚脐到耻骨之间的下腹；前后=下腹圆润度", axes="YZ", region=["waist", "hip"], radius=7, parent="pelvis_center",
           owners={"Y": ["pelvis_bulge"]}),
    ]
    return D


def _navel_center(obj, rc):
    """navel = the vertices the MPFB navel target moves most (measured on the neutral body)"""
    st0 = C.get_state(obj)
    off0 = A.P.off
    A.P.off = {}
    try:
        st = C.default_state()
        st["navel_io"] = 0.8
        C.apply_state(obj, st)
        d = np.linalg.norm(C.raw_coords(obj) - rc, axis=1)
    finally:
        A.P.off = off0
        C.apply_state(obj, st0)
    return np.argsort(-d)[:12]


def _anchors(obj, rc, zone, mv):
    ZM = lambda *k: np.isin(zone, [C.ZID[x] for x in k])
    keep = zone != 0
    x, y, z = rc[:, 0], rc[:, 1], rc[:, 2]
    left = x > 0.004
    zmin = float(z[keep].min())
    ea, eb = A._adjacency(obj)

    def vg(n):
        return C.group_indices(obj, n)

    def pick(mask, key, k=3):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise RuntimeError("white point: empty selection")
        return idx[np.argsort(key[idx])[:k]]

    def ring(a, b):
        m = np.zeros(len(rc), bool)
        m[ea[a[ea] & b[eb]]] = True
        m[eb[a[eb] & b[ea]]] = True
        return m & a

    jc = lambda n: rc[vg(n)].mean(0)
    sh, el, wr, hip, kn, an = (jc("joint-l-" + n) for n in ("shoulder", "elbow", "hand", "upper-leg", "knee", "ankle"))
    nip = rc[vg("nipple")]
    nip = nip[nip[:, 0] > 0].mean(0)

    def seg(a, b, zones, t0, t1):
        ax = b - a
        t = (rc - a) @ ax / float(ax @ ax)
        return ZM(*zones) & left & (t > t0) & (t < t1)

    S = {}
    S["w_deltoid_l"] = pick(ZM("uarm") & left & (z > sh[2] - 0.06) & (np.linalg.norm(rc - sh, axis=1) < 0.10), -x, 4)
    arm_ring = ring(ZM("ribcage", "bust") & left, ZM("uarm") & left)
    S["w_armpit_f_l"] = pick(arm_ring, y, 3)
    S["w_armpit_b_l"] = pick(arm_ring, -y, 3)
    S["w_armpit_l"] = pick(arm_ring, z, 3)
    ua = seg(sh, el, ["uarm"], 0.35, 0.65)
    S["w_acromion_l"] = pick(ZM("uarm", "ribcage", "neck") & left & (np.abs(x - sh[0]) < 0.045) & (np.abs(y - sh[1]) < 0.04), -z, 3)
    S["w_trap_l"] = pick(ZM("ribcage", "uarm", "neck") & left & (np.abs(x - 0.10) < 0.02) & (np.abs(y - sh[1]) < 0.05) & (z > sh[2] - 0.01), -z, 3)
    back = ZM("ribcage", "uarm", "neck") & left
    S["w_scap_l"] = pick(back & (np.abs(x - 0.11) < 0.03) & (np.abs(z - (sh[2] - 0.07)) < 0.03), -y, 3)
    S["w_back_up_l"] = pick(back & (np.abs(x - 0.04) < 0.02) & (np.abs(z - (sh[2] - 0.05)) < 0.03), -y, 3)
    S["w_uarm_out_l"] = pick(ua, -x, 4)
    S["w_uarm_in_l"] = pick(ua, x, 4)
    la = seg(el, wr, ["larm"], 0.35, 0.65)
    S["w_larm_out_l"] = pick(la, -x, 4)
    S["w_larm_in_l"] = pick(la, x, 4)
    bz = ZM("bust") & left
    S["w_bust_in_l"] = pick(bz & (np.abs(z - nip[2]) < 0.03), x, 3)
    S["w_bust_out_l"] = pick(bz & (np.abs(z - nip[2]) < 0.03), -x, 3)
    S["w_bust_up_l"] = pick(bz & (np.abs(x - nip[0]) < 0.03), -z, 3)
    S["w_bust_low_l"] = pick(bz & (np.abs(x - nip[0]) < 0.03), z, 3)
    gz = ZM("glute") & left
    gmin = float(z[gz].min())
    peak = np.where(gz)[0][np.argmax(y[gz])]
    px, py, pz = rc[peak]
    S["w_glute_peak_l"] = np.where(gz & (np.linalg.norm(rc - rc[peak], axis=1) < 0.018))[0]
    col = gz & (np.abs(x - px) < 0.03) & (y > py - 0.05)
    S["w_glute_top_l"] = pick(col, -z, 3)
    S["w_glute_low_l"] = pick(col, z, 3)
    S["w_glute_out_l"] = pick(ZM("glute", "hip") & left & (np.abs(z - pz) < 0.03), -x, 3)
    S["w_glute_leg_l"] = pick(ZM("thigh") & left & (z < gmin + 0.03) & (z > gmin - 0.07) & (np.abs(x - px) < 0.05), -y, 3)
    ta = seg(hip, kn, ["thigh"], 0.40, 0.65)
    S["w_thigh_out_l"] = pick(ta, -x, 4)
    S["w_thigh_in_l"] = pick(ta, x, 4)
    S["w_knee_f_l"] = pick(ZM("thigh", "calf") & left & (np.abs(z - kn[2]) < 0.025), y, 3)
    ca = seg(kn, an, ["calf"], 0.30, 0.55)
    S["w_calf_out_l"] = pick(ca, -x, 4)
    S["w_calf_in_l"] = pick(ca, x, 4)
    S["w_calf_b_l"] = pick(ca, -y, 4)
    S["w_calf_in_up_l"] = pick(seg(kn, an, ["calf"], 0.10, 0.28), x, 4)
    S["w_calf_in_low_l"] = pick(seg(kn, an, ["calf"], 0.58, 0.78), x, 4)
    S["w_shin_l"] = pick(seg(kn, an, ["calf"], 0.28, 0.50), y, 4)
    # palm: width axis from the hand's own principal axes (the hand is not aligned with x)
    hz = np.where(ZM("hand") & left)[0]
    f3 = jc("joint-l-finger-3-1")
    palm = hz[np.linalg.norm(rc[hz] - 0.5 * (wr + f3), axis=1) < 0.045]
    c = rc[palm] - rc[palm].mean(0)
    _, vec = np.linalg.eigh(c.T @ c)
    axis_len = (f3 - wr) / np.linalg.norm(f3 - wr)
    cand = [vec[:, 0], vec[:, 1], vec[:, 2]]
    width = min(cand[1:], key=lambda v: abs(v @ axis_len)) if abs(cand[2] @ axis_len) < 0.8 else cand[1]
    thumb = jc("joint-l-finger-1-1") - f3
    if width @ thumb < 0:
        width = -width                                      # +width = thumb side
    pw = (rc - rc[palm].mean(0)) @ width
    S["w_palm_in_l"] = pick(np.isin(np.arange(len(rc)), palm), -pw, 4)
    S["w_palm_out_l"] = pick(np.isin(np.arange(len(rc)), palm), pw, 4)
    fz = ZM("foot") & left
    fy0, fy1 = float(y[fz].min()), float(y[fz].max())
    low = fz & (z < zmin + 0.05)
    mid = low & (np.abs(y - 0.5 * (fy0 + fy1)) < 0.03)
    S["w_foot_in_l"] = pick(mid, x, 3)
    S["w_foot_out_l"] = pick(mid, -x, 3)
    S["w_inst_l"] = pick(ZM("foot", "calf") & left & (np.abs(y - (an[1] - 0.04)) < 0.03) & (z < an[2] + 0.06), -z, 3)
    S["w_foot_fore_l"] = pick(low & (np.abs(y - (fy0 + 0.25 * (fy1 - fy0))) < 0.025), -x, 3)
    # midline (x ~ 0)
    nav_ids = _navel_center(obj, A.rest_co(obj))
    nav = rc[nav_ids].mean(0)
    mid_ok = (np.abs(x) < 0.012) & keep
    S["w_navel"] = nav_ids[np.abs(rc[nav_ids, 0]) < 0.02][:6]
    if len(S["w_navel"]) == 0:
        S["w_navel"] = pick(mid_ok & ZM("waist") & (np.abs(z - nav[2]) < 0.012), y, 3)
    under = float(z[ZM("bust") & left].min())                # the (enlarged) bust hangs close to the navel: put the waist point between them
    zw = nav[2] + max(0.015, 0.5 * (under - nav[2]))
    mid_t = (np.abs(x) < 0.0015) & keep                       # true centre-line vertices only: the point must sit at |x| < 2 mm (health "order" check)
    S["w_waist_front"] = pick(mid_t & ZM("waist") & (np.abs(z - zw) < 0.008), y, 3)
    S["w_lower_abd"] = pick(mid_t & ZM("waist", "hip") & (np.abs(z - (nav[2] - 0.075)) < 0.012), y, 3)
    return S


def build(obj):
    """-> list of fully anchored white specs.  Anchors are chosen on the CURRENT (confirmed stage-A) surface - the neutral MPFB body has a
    much smaller bust/hip, so neutral-surface anchors would land on the wrong place of the user's shape - and are stored as vertex ids,
    so they follow every later edit.  `rest` keeps the neutral coordinate of the same vertices (hop-distance weights use the neutral body)."""
    rc = A.rest_co(obj)
    co = C.raw_coords(obj)
    zone = C.load_zones(obj)
    mv = np.array(list(obj["cs_mvmap"]), dtype=np.int32)
    S = _anchors(obj, co, zone, mv)
    specs = []
    for sp in definitions():
        pid = sp["id"]
        if sp["vg"]:
            ids = np.concatenate([C.group_indices(obj, n) for n in sp["vg"]])
        elif pid in S:
            ids = S[pid]
        elif pid.endswith("_r") and pid[:-2] + "_l" in S:
            ids = mv[S[pid[:-2] + "_l"]]
        else:
            raise RuntimeError("no anchors for " + pid)
        sp["verts"] = [int(i) for i in np.unique(ids)]
        sp["rest"] = [round(float(v), 5) for v in rc[sp["verts"]].mean(0)]
        sp["mirror"] = (pid[:-2] + ("_r" if pid.endswith("_l") else "_l")) if sp["side"] in ("L", "R") else None
        specs.append(sp)
    return specs


def add_to_object(obj):
    """one-off upgrade of a stage-A file (run it on the state that is the stage-B baseline): append the white points to `cs_points` (green points / offsets untouched). Idempotent."""
    reg = json.loads(obj["cs_points"])
    if any(p.get("white") for p in reg):
        return 0
    specs = build(obj)
    obj["cs_points"] = json.dumps(reg + specs)
    A.load(obj)
    return len(specs)


MAIN_UPGRADE = ["w_uarm_in_", "w_larm_in_", "w_armpit_f_", "w_armpit_b_", "w_thigh_in_", "w_calf_in_"]      # inner-side points: now shown by default (v018)


def extend_object(obj):
    """v018 upgrade of a stage-B file: append the white points that are not registered yet (anchors picked on the CURRENT surface) and
    make the inner-side points visible by default.  Existing points / offsets untouched.  Returns (added ids, main-flag changes)."""
    reg = json.loads(obj["cs_points"])
    have = {p["id"] for p in reg}
    new = [sp for sp in build(obj) if sp["id"] not in have]
    changed = 0
    for p in reg:
        if p.get("white") and any(p["id"].startswith(k) for k in MAIN_UPGRADE) and not p["main"]:
            p["main"] = True
            changed += 1
    obj["cs_points"] = json.dumps(reg + new)
    A.load(obj)
    return [sp["id"] for sp in new], changed


def is_white(p):
    return bool(p.get("white"))
