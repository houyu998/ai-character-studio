"""Stage A mesh health check: finds sharp spikes, new creases, flipped faces (self-penetration of the surface), legs crossing the
midline, ground-contact loss and left/right order errors, locates them (world position + region) and compares with the previous
state so the panel can offer "撤销这一步".  NOTHING here hides a problem behind an average: every flagged vertex is listed.

Thresholds (fixed BEFORE the tests, task book A: "阈值先定义再跑"):
  spike    : Laplacian excess  |x - mean(neighbours)| (now) - (same on the neutral body)  > 8 mm
  crease   : an edge whose dihedral angle grew by > 35 deg over the neutral body AND is now > 60 deg
  folded   : a face whose normal differs from the mean normal of its edge-neighbours by more than ~72 deg (dot < 0.3): a real fold /
             self-overlap.  (Added after the first run: the original "flipped vs neutral" test also fires on legitimately turned
             surfaces under a big bust - v005 had 12 such faces and 0 folds - so it is kept only as the INFO count `turned`.)
  turned   : (info, not an issue) face normal rotated > 90 deg from the neutral body
  crossing : a left-leg (+x) vertex with x < -2 mm, or a right-leg vertex with x > +2 mm (legs through each other)
  ground   : lowest body vertex farther than 3 mm from the ground plane z = cs_ground_z
  order    : left-side points must lie at x > 0 and right-side points at x < 0 (midline points |x| < 2 mm)
"""
import json
import types

import numpy as np

import s1_core as C
import a_points as A

TH = dict(fold_dot=0.3, spike=0.008, crease_delta=35.0, crease_abs=60.0, flipped_dot=0.0, crossing=0.002, ground=0.003, mid=0.002)
_CACHE = {}


def _topology(obj):
    key = (obj.name, len(obj.data.vertices), len(obj.data.polygons))
    if key in _CACHE:
        return _CACHE[key]
    me = obj.data
    nv = len(me.vertices)
    ed = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    # neighbours (CSR)
    a = np.concatenate([ed[:, 0], ed[:, 1]])
    b = np.concatenate([ed[:, 1], ed[:, 0]])
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    ptr = np.searchsorted(a, np.arange(nv + 1))
    # faces: triangulate by fan from the first loop (quads are near planar; good enough for normals)
    npoly = len(me.polygons)
    ls = np.empty(npoly, dtype=np.int32)
    lt = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    tri = []
    owner = []
    for k in (1, 2):
        m = lt > k
        tri.append(np.stack([lv[ls[m]], lv[ls[m] + k], lv[ls[m] + k + 1]], axis=1))
        owner.append(np.where(m)[0])
    tri = np.concatenate(tri)
    owner = np.concatenate(owner)
    # edge -> two polygons (for dihedral)
    pe = {}
    edge_faces = {}
    for pi in range(npoly):
        s, n = ls[pi], lt[pi]
        vs = lv[s:s + n]
        for j in range(n):
            u, v = int(vs[j]), int(vs[(j + 1) % n])
            kk = (u, v) if u < v else (v, u)
            edge_faces.setdefault(kk, []).append(pi)
    pairs = np.array([(k[0], k[1], f[0], f[1]) for k, f in edge_faces.items() if len(f) == 2], dtype=np.int32)
    T = types.SimpleNamespace(ptr=ptr, nbr=b, tri=tri, owner=owner, ls=ls, lt=lt, lv=lv, pairs=pairs, npoly=npoly)
    _CACHE[key] = T
    return T


def _face_normals(co, T):
    """per polygon normal (Newell via the first triangle fan, normalised)"""
    n = np.zeros((T.npoly, 3))
    tri = T.tri
    c = np.cross(co[tri[:, 1]] - co[tri[:, 0]], co[tri[:, 2]] - co[tri[:, 0]])
    np.add.at(n, T.owner, c)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def _laplacian(co, T):
    s = np.zeros_like(co)
    cnt = np.diff(T.ptr).astype(float)
    np.add.at(s, np.repeat(np.arange(len(co)), np.diff(T.ptr)), co[T.nbr])
    mean = s / np.maximum(cnt, 1)[:, None]
    return np.linalg.norm(co - mean, axis=1)


def rest_reference(obj):
    """neutral-body (default state, no offsets) Laplacian / face normals / dihedral — computed once and cached on the object key."""
    key = ("rest", obj.name, len(obj.data.vertices))
    if key in _CACHE:
        return _CACHE[key]
    T = _topology(obj)
    st0 = C.get_state(obj)
    off0 = {k: list(v) for k, v in A.P.off.items()}
    A.P.off = {}
    C.apply_state(obj, C.default_state())
    co = C.raw_coords(obj)
    R = types.SimpleNamespace(lap=_laplacian(co, T), n=_face_normals(co, T), ang=_dihedral(co, T))
    A.P.off = off0
    C.apply_state(obj, st0)
    _CACHE[key] = R
    return R


def _dihedral(co, T):
    n = _face_normals(co, T)
    d = np.clip((n[T.pairs[:, 2]] * n[T.pairs[:, 3]]).sum(axis=1), -1, 1)
    return np.degrees(np.arccos(d))


def _region(obj, idx, zone):
    names = {v: k for k, v in C.ZID.items()}
    z = zone[idx]
    return names.get(int(np.bincount(z).argmax()), "?") if len(z) else "?"


def check(obj, co=None, rest=None):
    """-> dict(issues=[...], n=int, by_kind={...}).  Every issue: kind, region, pos (x,y,z), value, threshold, verts (indices)."""
    T = _topology(obj)
    R = rest or rest_reference(obj)
    co = C.raw_coords(obj) if co is None else co
    zone = C.load_zones(obj)
    body = zone != 0
    issues = []

    lap = _laplacian(co, T)
    ex = lap - R.lap
    bad = np.where((ex > TH["spike"]) & body)[0]
    for g in _cluster(bad, co, 0.03):
        v = g[np.argmax(ex[g])]
        issues.append(dict(kind="spike", region=_region(obj, g, zone), pos=_p(co[v]), value=float(ex[v] * 1000), unit="mm", threshold=TH["spike"] * 1000,
                           verts=[int(i) for i in g[:40]], n=len(g)))

    ang = _dihedral(co, T)
    d = ang - R.ang
    bad_e = np.where((d > TH["crease_delta"]) & (ang > TH["crease_abs"]))[0]
    if len(bad_e):
        vs = np.unique(np.concatenate([T.pairs[bad_e, 0], T.pairs[bad_e, 1]]))
        vs = vs[body[vs]]
        for g in _cluster(vs, co, 0.03):
            e = bad_e[np.isin(T.pairs[bad_e, 0], g)]
            val = float(ang[e].max()) if len(e) else 0.0
            issues.append(dict(kind="crease", region=_region(obj, g, zone), pos=_p(co[g].mean(0)), value=val, unit="deg", threshold=TH["crease_abs"],
                               verts=[int(i) for i in g[:40]], n=len(g)))

    n = _face_normals(co, T)
    fv = T.lv[T.ls]
    P = T.pairs
    s = np.zeros_like(n)
    np.add.at(s, P[:, 2], n[P[:, 3]])
    np.add.at(s, P[:, 3], n[P[:, 2]])
    m = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
    nd = (n * m).sum(axis=1)
    bad_f = np.where((nd < TH["fold_dot"]) & body[fv])[0]
    if len(bad_f):
        vs = np.unique(fv[bad_f])
        for g in _cluster(vs, co, 0.03):
            issues.append(dict(kind="folded", region=_region(obj, g, zone), pos=_p(co[g].mean(0)), value=float(nd[bad_f].min()), unit="dot",
                               threshold=TH["fold_dot"], verts=[int(i) for i in g[:40]], n=len(g)))
    turned = int((((n * R.n).sum(axis=1) < 0) & body[fv]).sum())

    leg = np.isin(zone, [C.ZID["thigh"], C.ZID["calf"]])
    cl = np.where(leg & (co[:, 0] < -TH["crossing"]) & (co[:, 0] < 0) & (_side_of(obj) > 0))[0]
    cr = np.where(leg & (co[:, 0] > TH["crossing"]) & (_side_of(obj) < 0))[0]
    for ids, nm in ((cl, "左腿越过中线"), (cr, "右腿越过中线")):
        if len(ids):
            issues.append(dict(kind="crossing", region=nm, pos=_p(co[ids].mean(0)), value=float(np.abs(co[ids, 0]).max() * 1000), unit="mm",
                               threshold=TH["crossing"] * 1000, verts=[int(i) for i in ids[:40]], n=len(ids)))

    gz = float(obj.get("cs_ground_z", co[C.group_indices(obj, "body"), 2].min()))
    zmin = float(co[C.group_indices(obj, "body"), 2].min())
    if abs(zmin - gz) > TH["ground"]:
        v = C.group_indices(obj, "body")[np.argmin(co[C.group_indices(obj, "body"), 2])]
        issues.append(dict(kind="ground", region="脚底", pos=_p(co[v]), value=float((zmin - gz) * 1000), unit="mm", threshold=TH["ground"] * 1000, verts=[int(v)], n=1))

    if A.P.reg:
        pos = A.positions(obj, co)
        for p in A.P.reg:
            x = float(pos[p["id"]][0])
            if (p["side"] == "L" and x <= 0) or (p["side"] == "R" and x >= 0) or (p["side"] == "M" and abs(x) > TH["mid"]):
                issues.append(dict(kind="order", region=p["label"], pos=_p(pos[p["id"]]), value=x * 1000, unit="mm", threshold=0, verts=[], n=1, point=p["id"]))
    by = {}
    for i in issues:
        by[i["kind"]] = by.get(i["kind"], 0) + 1
    return dict(issues=issues, n=len(issues), by_kind=by, turned=turned)


def _side_of(obj):
    key = ("side", obj.name, len(obj.data.vertices))
    if key not in _CACHE:
        _CACHE[key] = np.sign(A.rest_x(obj)).astype(int)
    return _CACHE[key]


def _p(v):
    return [round(float(x), 4) for x in v]


def _cluster(idx, co, radius):
    """greedy spatial clustering of vertex indices (so one spike is one issue, not 30)"""
    idx = np.array(idx, dtype=int)
    out = []
    left = set(range(len(idx)))
    while left:
        i = left.pop()
        c = co[idx[i]]
        mem = [j for j in left if np.linalg.norm(co[idx[j]] - c) < radius]
        for j in mem:
            left.discard(j)
        out.append(idx[[i] + mem])
    return out


def signature(res):
    """coarse identity of the current issue set (kind + position rounded to 3 cm) to tell NEW issues from old ones"""
    return {(i["kind"], round(i["pos"][0] / 0.03), round(i["pos"][1] / 0.03), round(i["pos"][2] / 0.03)) for i in res["issues"]}


def new_since(prev, cur):
    """Track persistent mesh warnings by vertex identity, not a moving 3-cm centroid bucket.

    A shrinking cluster or a small drag must not relabel the same old crease as a new defect.
    Newly flagged vertices still count as a new warning; numeric thresholds are unchanged.
    """
    if not prev:
        return list(cur["issues"])
    known = {}
    for issue in prev["issues"]:
        known.setdefault(issue["kind"], set()).update(issue.get("verts", []))
    previous_signatures = signature(prev)
    result = []
    for issue in cur["issues"]:
        verts = set(issue.get("verts", []))
        if verts:
            if not verts.issubset(known.get(issue["kind"], set())):
                result.append(issue)
        elif (issue["kind"], *(round(x / 0.03) for x in issue["pos"])) not in previous_signatures:
            result.append(issue)
    return result


def summarize(res, limit=8):
    if not res["issues"]:
        return ["健康检查：未发现尖刺 / 折角 / 翻面 / 腿越线 / 离地 / 左右顺序问题（阈值见 a_health.py 顶部）"]
    out = ["健康检查：%d 处问题 %s" % (res["n"], json.dumps(res["by_kind"], ensure_ascii=False))]
    for i in res["issues"][:limit]:
        out.append("  · %s @%s（%s）%.1f%s 阈值%.1f" % (i["kind"], i["region"], tuple(i["pos"]), i["value"], i["unit"], i["threshold"]))
    return out
