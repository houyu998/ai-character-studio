"""v3 circumferences from REAL closed cross-sections (A3: "围度来自真实闭合截面").

The mesh is cut exactly by a horizontal plane (edge/plane intersections, same as s1_core.Profiler); the circumference
is the TAPE-MEASURE length of that closed section = perimeter of its convex hull (a tape bridges the cleavage / the
spine furrow, like a real tape measure).  Arms / hands / helpers are excluded for torso girths; the thigh girth uses
the left leg only (x > 0) below the crotch.  Lengths in mm."""
import numpy as np

import s1_core as C


def hull_perimeter(p):
    """convex hull perimeter of 2-D points (monotone chain)"""
    p = np.unique(np.round(np.asarray(p, dtype=np.float64), 7), axis=0)
    if len(p) < 3:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, up = half(p), half(p[::-1])
    hull = np.array(lo[:-1] + up[:-1])
    return float(np.linalg.norm(np.roll(hull, -1, axis=0) - hull, axis=1).sum())


class Girths:
    def __init__(self, obj, zone):
        self.obj, self.zone = obj, zone
        self.torso = C.Profiler(obj, zone, exclude=("uarm", "larm", "hand", "helper", "head"))
        self.leg = C.Profiler(obj, zone, exclude=("uarm", "larm", "hand", "helper", "head", "ribcage", "bust"))

    def section(self, co, z, which="torso"):
        q = (self.torso if which == "torso" else self.leg).slice(co, z)
        if which == "thigh":
            q = q[q[:, 0] > 0.0]
        return q

    def girth(self, co, z, which="torso"):
        return hull_perimeter(self.section(co, z, which)) * 1000.0

    def __call__(self, co):
        z = self.zone
        Z = C.ZID
        body = z != Z["helper"]
        zmin = float(co[body, 2].min())
        H = float(np.ptp(co[body, 2]))
        nip = C.group_indices(self.obj, "nipple")
        z_bust = float(co[nip, 2].mean()) if len(nip) else zmin + 0.72 * H
        bz = co[(z == Z["bust"]), 2]
        z_under = float(np.percentile(bz, 3)) if len(bz) else z_bust - 0.08
        mid = body & (np.abs(co[:, 0]) < 0.004) & np.isin(z, [Z["hip"], Z["glute"], Z["thigh"]])
        z_crotch = float(co[mid, 2].min()) if mid.any() else zmin + 0.47 * H
        out = dict(H=H * 1000.0, bust=self.girth(co, z_bust), underbust=self.girth(co, z_under - 0.005))
        zs = np.linspace(z_crotch + 0.12, z_under - 0.02, 24)                  # natural waist = narrowest
        g = [self.girth(co, x) for x in zs]
        i = int(np.argmin(g)); out["waist"] = g[i]; z_waist = zs[i]
        zs = np.linspace(z_crotch + 0.03, z_waist - 0.04, 24)                  # hip = widest below the waist
        g = [self.girth(co, x) for x in zs]
        i = int(np.argmax(g)); out["hip"] = g[i]
        zs = np.linspace(z_crotch - 0.10, z_crotch - 0.02, 9)                  # upper thigh (left)
        out["thigh"] = max(self.girth(co, x, "thigh") for x in zs)
        out["z"] = dict(bust=z_bust, underbust=z_under, waist=float(z_waist), crotch=z_crotch)
        return out
