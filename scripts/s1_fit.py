"""Stage-1: fit MPFB's EXISTING bounded controls to the reference silhouette numbers and report the residual gap.

Run:  blender -b --factory-startup --python s1_fit.py -- [out_json]
Nothing here creates new deformations; anything the bounded controls cannot reach is listed as a gap
(that is what the trial copy in s1_custom.py is for).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
import s1_core as C

BASE_MACRO = {"gender": 0.0, "age": 0.5, "muscle": 0.5, "weight": 0.5, "proportions": 0.5, "height": 0.5,
              "cupsize": 0.5, "firmness": 0.5, "race": {"asian": 1.0, "caucasian": 0.0, "african": 0.0}}

# (fl, quantity, weight, why).  quantities: front_w / leg_w / side_fwd / side_back  (fractions of H)
# plus landmark terms ("crotch", "knee", "nipple") given as fl of the landmark.
FIT_TERMS = [
    # hips / pelvis (above the crotch, stance-free)
    (0.525, "front_w", 2.0, "胯宽"), (0.55, "front_w", 1.5, "胯宽"), (0.575, "front_w", 1.5, "胯-腰过渡"),
    (0.60, "front_w", 1.5, "腰下"), (0.625, "front_w", 1.5, "腰"), (0.65, "front_w", 2.0, "腰最细"),
    (0.70, "front_w", 0.5, "下胸廓"), (0.725, "front_w", 0.5, "胸宽"),
    # legs, per leg (stance-free)
    (0.10, "leg_w", 0.5, "踝"), (0.20, "leg_w", 1.0, "小腿"), (0.25, "leg_w", 1.0, "小腿"),
    (0.30, "leg_w", 1.0, "膝"), (0.35, "leg_w", 1.0, "膝上"), (0.40, "leg_w", 1.5, "大腿"),
    (0.45, "leg_w", 1.5, "大腿"), (0.50, "leg_w", 1.0, "大腿根"),
    # side view: pelvis / glute / belly / bust
    (0.50, "side_back", 1.5, "臀后凸"), (0.525, "side_back", 2.0, "臀后凸"), (0.55, "side_back", 2.0, "臀后凸"),
    (0.575, "side_back", 1.0, "臀上缘"), (0.60, "side_back", 1.0, "腰后"),
    (0.45, "side_back", 1.0, "臀下缘"),
    (0.50, "side_fwd", 1.0, "下腹前"), (0.55, "side_fwd", 1.0, "下腹前"), (0.60, "side_fwd", 1.0, "腹前"),
    (0.65, "side_fwd", 1.0, "腰前"),
    (0.70, "side_fwd", 1.5, "胸前"), (0.725, "side_fwd", 3.0, "胸前最凸"), (0.75, "side_fwd", 1.5, "胸前"),
    (0.40, "side_fwd", 0.7, "大腿前"), (0.25, "side_back", 0.7, "小腿后"),
    # v002 additions (from the independent per-row self-check of v001, see docs/S1_test_log.md):
    # leg position in the front view (outer / inner edge of one leg, stance-DEPENDENT: covers knees together)
    *[(f, "in_half", 1.0, "腿内缘/膝并拢") for f in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.475)],
    *[(f, "out_half", 1.0, "腿外缘") for f in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)],
    # leg line in the side view (knee sits behind the ankle-hip line in the reference)
    *[(f, "side_fwd", 1.0, "腿线前缘") for f in (0.20, 0.25, 0.30, 0.35, 0.45)],
    *[(f, "side_back", 1.0, "腿线后缘") for f in (0.20, 0.30, 0.35, 0.40)],
    # lower glute / lumbar back.  side_back at >= 0.625 H is NOT used: the reference's long hair hangs behind the back.
    # Evidence (v008 diag): dynamic ref side_back goes 0.6->0.016, 0.625->0.0412, 0.65->0.0616 - a physically impossible
    # dip-then-bulge for a lower back; the clean STATIC ref is 0.022 at 0.625 H.  Chasing 0.0412 was v007's worst
    # "residual" (-2.66 %H) and is a hair target, not the body - so it is dropped (see docs/阶段A_问题清单.md P-back).
    (0.475, "side_back", 1.0, "下臀"),
    # landmarks (fl of the landmark)
    ("crotch", None, 4.0, "裆高（腿长比）"),
    ("knee", None, 2.0, "膝高"),
    ("nipple", None, 2.0, "胸点高"),
]
LANDMARK_REF = {"crotch": 0.516, "knee": 0.295, "nipple": 0.725}
# hard bounds used by the fit (the panel itself allows the full MPFB range; the DRAFT stays in a sane range)
FIT_BOUND_BI = 0.6
FIT_BOUNDS = {"height": (0.5, 0.60), "proportions": (0.5, 0.75), "weight": (0.40, 0.60), "muscle": (0.40, 0.60),
              "cupsize": (0.5, 1.0)}
H_MAX = 1.72               # metres; absolute height is unknown from the image (category B), keep it plausible
BOUND_OVERRIDE = {}        # per-run overrides from the command line, e.g. {"hip_h": 0.8}
# controls whose target cannot be seen in a silhouette (set by eye from the shaded reference, then fixed during the fit)
FIXED_DEFAULT = {"bust_pos": 0.5, "bust_dist": -0.7, "bust_vol_v": 0.5}


def bounds(c):
    if c["id"] in BOUND_OVERRIDE:
        b = BOUND_OVERRIDE[c["id"]]
        return (-b, b) if c["kind"] == "bi" else (c["lo"], b)
    if c["id"] in FIT_BOUNDS:
        return FIT_BOUNDS[c["id"]]
    return (-FIT_BOUND_BI, FIT_BOUND_BI) if c["kind"] == "bi" else (c["lo"], c["hi"])


class Fitter:
    def __init__(self, obj, zone):
        self.obj = obj
        self.zone = zone
        co0 = C.raw_coords(obj)
        self.knee_ids = C.group_indices(obj, "joint-l-knee")
        self.body = C.group_indices(obj, "body")
        self.nip_ids = C.group_indices(obj, "nipple")
        self.neval = 0
        self.prof = C.Profiler(obj, zone)

    def preload(self):
        for c in C.CONTROLS:
            if c["kind"] != "bi":
                continue
            for n in c["incr"] + c["decr"]:
                if n not in (self.obj.data.shape_keys.key_blocks if self.obj.data.shape_keys else {}):
                    path = C.TargetService.target_full_path(n)
                    C.TargetService.load_target(self.obj, path, weight=0.0, name=n)

    def measure(self, state):
        C.apply_state(self.obj, state)
        co = C.raw_coords(self.obj)
        mp = self.prof(co)
        H = mp["H"]
        zmin = mp["zmin"]
        z = co[:, 2]
        mid = np.zeros(len(co), bool)
        mid[self.body] = True
        zmax = z[self.body].max()
        mid &= (np.abs(co[:, 0]) < 0.012) & (z > 0.35 * zmax) & (z < 0.62 * zmax)
        lm = {"crotch": (z[mid].min() - zmin) / H,
              "knee": (z[self.knee_ids].mean() - zmin) / H,
              "nipple": (z[self.nip_ids].mean() - zmin) / H}
        self.neval += 1
        return mp, lm

    def rows(self, mp, lm):
        out = []
        for fl, q, w, why in FIT_TERMS:
            if q is None:
                lref = (C.REF_OVERRIDE or {}).get("landmarks", LANDMARK_REF) if C.REF_OVERRIDE is not None else LANDMARK_REF
                if fl not in lref:
                    continue
                out.append((fl, "landmark", w, why, lref[fl], lm[fl]))
            else:
                p = mp["profile"].get(fl)
                if p is None:
                    continue
                rv = C.ref_value(fl, q)
                if rv is None:
                    continue
                out.append((fl, q, w, why, rv, p[q]))
        return out

    def objective(self, state, lam=0.004):
        mp, lm = self.measure(state)
        tot = 0.0
        for fl, q, w, why, ref, val in self.rows(mp, lm):
            # absolute error in units of 1% of H  (so 1.0 == 1% body height)
            e = (val - ref) / 0.01
            tot += w * e * e
        full = C.clamp_state(state)
        for c in C.CONTROLS:
            if c["kind"] == "bi":
                v = abs(full.get(c["id"], 0.0))
                tot += lam * 100 * v * v
        if mp["H"] > H_MAX:
            tot += 4000 * (mp["H"] - H_MAX) ** 2 / 0.0001 * 0.0001 * 100
        return tot


def fit(fitter, state, free_ids, steps=(0.4, 0.2, 0.1, 0.05, 0.02), macro_steps=(0.15, 0.07, 0.03), sweeps=6, log=print):
    best = fitter.objective(state)
    log("start objective %.2f" % best)
    for step_i, st in enumerate(steps):
        for sw in range(sweeps):
            improved = False
            for cid in free_ids:
                c = C.CTRL[cid]
                s = macro_steps[min(step_i, len(macro_steps) - 1)] if c["kind"] == "macro" else st
                cur = state[cid]
                cand = []
                for d in (+s, -s):
                    lo, hi = bounds(c)
                    v = min(max(cur + d, lo), hi)
                    if abs(v - cur) > 1e-9:
                        cand.append(v)
                bestv, bestobj = cur, best
                for v in cand:
                    trial = dict(state)
                    trial[cid] = v
                    o = fitter.objective(trial)
                    if o < bestobj - 1e-6:
                        bestobj, bestv = o, v
                if bestv != cur:
                    state[cid] = bestv
                    best = bestobj
                    improved = True
            log("  step %.2f sweep %d objective %.2f" % (st, sw, best))
            if not improved:
                break
    return state, best


def build_base():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    obj = C.HumanService.create_human(mask_helpers=True, detailed_helpers=True, extra_vertex_groups=True,
                                      feet_on_ground=True, scale=0.1, macro_detail_dict=BASE_MACRO)
    return obj


def main(argv):
    """argv: [out_json] [--init state.json] [--fix k=v,k=v] [--bound k=v,k=v] [--nofit]
    --init  start from an earlier state (default: MPFB defaults); --fix  hold these controls at the given values
    --bound widen/narrow the symmetric bound of a bipolar control for this run (slider range is 0.8)"""
    import s1_custom as U
    C.enable_mpfb()
    U.register()
    pos = [a for a in argv if not a.startswith("--")]
    opt = {}
    for i, a in enumerate(argv):
        if a.startswith("--"):
            opt[a[2:]] = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
    out_json = pos[0] if pos else os.path.join(C.VERS, "s1_fit_result.json")
    obj = build_base()
    co = C.raw_coords(obj)
    zone = C.compute_zones(obj, co)
    C.store_zones(obj, zone)
    F = Fitter(obj, zone)
    F.preload()
    state = C.default_state()
    if "init" in opt:
        init = json.load(open(opt["init"], encoding="utf-8"))
        state.update(init.get("state", init))
    fixed = dict(FIXED_DEFAULT)
    if "fix" in opt:
        fixed = {k: float(v) for k, v in (p.split("=") for p in opt["fix"].split(","))} if opt["fix"] != "none" else {}
    if "bound" in opt:
        BOUND_OVERRIDE.update({k: float(v) for k, v in (p.split("=") for p in opt["bound"].split(","))})
    state.update(fixed)
    mp0, lm0 = F.measure(state)
    rows0 = F.rows(mp0, lm0)
    obj0 = F.objective(state)
    t = time.time()
    # free controls: stage-1 MPFB controls + the trial ones, minus the ones fixed by eye
    skip = ("firmness", "head_scale_h", "uarm_len", "larm_len", "uarm_c") + tuple(fixed)
    free = [c["id"] for c in C.CONTROLS if c["stage"] in ("①", "①试") and c["id"] not in skip]
    if "nofit" in opt:
        free = []
    print("free controls:", len(free), free)
    print("fixed by eye:", fixed)
    state, best = fit(F, state, free)
    print("fit done in %.1fs, %d evaluations, objective %.2f (baseline %.2f)" % (time.time() - t, F.neval, best, obj0))
    mp1, lm1 = F.measure(state)
    rows1 = F.rows(mp1, lm1)
    res = dict(state=state, objective=best, baseline_objective=obj0,
               rows=[dict(fl=r[0], q=r[1], w=r[2], why=r[3], ref=r[4], base=b[5], fit=r[5]) for r, b in zip(rows1, rows0)],
               H=mp1["H"])
    json.dump(res, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("%-8s %-10s %-10s %8s %8s %8s" % ("fl", "quantity", "why", "ref", "base", "fit"))
    for r in res["rows"]:
        print("%-8s %-10s %-10s %8.3f %8.3f %8.3f  err(fit)=%+.3f" % (r["fl"], r["q"], r["why"], r["ref"], r["base"], r["fit"], r["fit"] - r["ref"]))
    print("state (non-default):")
    for k, v in state.items():
        if abs(v - C.CTRL[k]["default"]) > 1e-6:
            print("  %-14s %+.3f" % (k, v))
    print("height m:", mp1["H"])


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
