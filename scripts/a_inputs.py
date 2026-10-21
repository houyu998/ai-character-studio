"""Stage A inputs: reference images (single/multi, view tags, replace/remove, show toggle, in-project copies), prompt
understanding ("规则指令模式": vocabulary rules, NOT a language model), no-image creation with presets (built-in + saved), the
reference/prompt -> draft generators, and the entry for free-form AI (marked 未完成: no authorised AI environment).

Everything is plain data stored on the body object (cs_refs / cs_ref_metrics / cs_understanding / cs_prompt) and as files under
reference/inputs, presets/, docs/AI_inbox.md, so a saved .blend re-opens with its inputs.
"""
import os
import re
import sys
import json
import time
import math
import shutil
import subprocess

import bpy
import numpy as np

import s1_core as C
import s1_fit as F
import a_points as A

ROOT = C.ROOT
INPUTS = os.path.join(ROOT, "reference", "inputs")
PRESETS = os.path.join(ROOT, "presets")
USER_PRESETS = os.path.join(PRESETS, "user")
INBOX = os.path.join(ROOT, "docs", "AI_inbox.md")
ANALYZER = os.path.join(ROOT, "scripts", "a_imganalyze.py")
PLANE = 0.9                       # distance (m) of the image planes from the body axis
VIEWS = ("front", "side", "back", "q34", "other")
HEIGHT_M_PER_UNIT = 1.33
HEIGHT_M_AT_HALF = 1.496


# =========================================================== reference images
def _py():
    for cand in ("python", "py"):
        p = shutil.which(cand)
        if p:
            return p
    raise RuntimeError("找不到系统 Python（图像分析需要 numpy/scipy/PIL）")


def refs(obj):
    return json.loads(obj.get("cs_refs", "[]"))


def save_refs(obj, lst):
    obj["cs_refs"] = json.dumps(lst, ensure_ascii=False)


def active_refs(obj):
    return [r for r in refs(obj) if not r.get("removed")]


def _unique_dest(name):
    os.makedirs(INPUTS, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(name))
    dest = os.path.join(INPUTS, stem + ext)
    n = 1
    while os.path.exists(dest):
        n += 1
        dest = os.path.join(INPUTS, "%s_%d%s" % (stem, n, ext))
    return dest


def _run_analyzer(items, out_json):
    man = out_json.replace(".json", "_manifest.json")
    json.dump(dict(images=items), open(man, "w", encoding="utf-8"), ensure_ascii=False)
    r = subprocess.run([_py(), ANALYZER, man, out_json], capture_output=True, text=True, timeout=180, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("图像分析失败：" + (r.stderr or r.stdout)[-400:])
    return json.load(open(out_json, encoding="utf-8"))


def _analysis_path(tag):
    os.makedirs(os.path.join(INPUTS, "_analysis"), exist_ok=True)
    return os.path.join(INPUTS, "_analysis", "%s_%s.json" % (tag, time.strftime("%Y%m%d_%H%M%S")))


def add_reference(obj, path, view="front", sheet_views=None, facing=None):
    """Copy the image into the project (originals are never touched) and register it.  view='sheet' splits a turnaround sheet
    into one reference per view.  Returns the list of new entries (with the analysis geometry)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    dest = _unique_dest(path)
    shutil.copy2(path, dest)
    new = []
    lst = refs(obj)
    if view == "sheet":
        sv = list(sheet_views or ["front", "side", "back"])
        res = _run_analyzer([dict(path=dest, view="sheet", sheet_views=sv, crop_dir=INPUTS)], _analysis_path("crop"))
        crops = None
        for n in res["notes"]:
            if n.startswith("SHEET_CROPS:"):
                crops = json.loads(n[len("SHEET_CROPS:"):])
            else:
                raise RuntimeError(n)
        for c in crops:
            new.append(dict(file=os.path.relpath(c["path"], ROOT), orig=path, view=c["view"], visible=True, added=time.strftime("%Y-%m-%d %H:%M:%S"),
                            parent=os.path.relpath(dest, ROOT)))
    else:
        new.append(dict(file=os.path.relpath(dest, ROOT), orig=path, view=view, visible=True, added=time.strftime("%Y-%m-%d %H:%M:%S"), facing=facing))
    nid = max([r.get("id", 0) for r in lst] + [0])
    for e in new:
        nid += 1
        e["id"] = nid
        lst.append(e)
    save_refs(obj, lst)
    _geometry(obj, [e["id"] for e in new])
    rebuild_empties(obj)
    return [r for r in refs(obj) if r["id"] in [e["id"] for e in new]]


def _geometry(obj, ids):
    """per-image figure geometry (height px, ground row, axis) used to scale/align the image planes."""
    lst = refs(obj)
    sel = [r for r in lst if r["id"] in ids]
    items = [dict(path=os.path.join(ROOT, r["file"]), view=r["view"] if r["view"] in ("front", "side", "back") else "front",
                  facing=r.get("facing")) for r in sel]
    res = _run_analyzer(items, _analysis_path("geom"))
    for r, v in zip(sel, res["views"]):
        r["geom"] = dict(H_px=v["H_px"], top=v["top"], bot=v["bot"], axis=v["axis"], w=v["img_w"], h=v["img_h"])
        if v.get("facing") and r["view"] == "side":
            r["facing"] = r.get("facing") or v["facing"]
    save_refs(obj, lst)


def remove_reference(obj, rid):
    """Unregister (the file stays in reference/inputs — nothing is deleted from disk)."""
    lst = refs(obj)
    for r in lst:
        if r["id"] == rid:
            r["removed"] = True
    save_refs(obj, lst)
    e = bpy.data.objects.get("CS_Ref_%d" % rid)
    if e is not None:
        bpy.data.objects.remove(e, do_unlink=True)


def replace_reference(obj, rid, path, view=None):
    """New copy in the project; the entry keeps its id/view unless a new view tag is given; the old copy stays on disk."""
    dest = _unique_dest(path)
    shutil.copy2(path, dest)
    lst = refs(obj)
    for r in lst:
        if r["id"] == rid:
            r["history"] = r.get("history", []) + [r["file"]]
            r["file"] = os.path.relpath(dest, ROOT)
            r["orig"] = path
            if view:
                r["view"] = view
    save_refs(obj, lst)
    _geometry(obj, [rid])
    e = bpy.data.objects.get("CS_Ref_%d" % rid)
    if e is not None:
        bpy.data.objects.remove(e, do_unlink=True)
    rebuild_empties(obj)


def set_view_tag(obj, rid, view):
    lst = refs(obj)
    for r in lst:
        if r["id"] == rid:
            r["view"] = view
    save_refs(obj, lst)
    _geometry(obj, [rid])
    e = bpy.data.objects.get("CS_Ref_%d" % rid)
    if e is not None:
        bpy.data.objects.remove(e, do_unlink=True)
    rebuild_empties(obj)


def set_visible(obj, on, rid=None):
    lst = refs(obj)
    for r in lst:
        if rid is None or r["id"] == rid:
            r["visible"] = bool(on)
    save_refs(obj, lst)
    for r in lst:
        e = bpy.data.objects.get("CS_Ref_%d" % r["id"])
        if e is not None and (rid is None or r["id"] == rid):
            e.hide_viewport = not on


def body_height(obj):
    co = C.raw_coords(obj)
    z = C.load_zones(obj)
    k = z != 0
    return float(co[k, 2].max() - co[k, 2].min()), float(co[k, 2].min())


def _plumb_y(obj):
    co = C.raw_coords(obj)
    H, z0 = body_height(obj)
    band = (co[:, 2] > z0 + 0.085 * H) & (co[:, 2] < z0 + 0.1 * H) & (C.load_zones(obj) != 0)
    return float(0.5 * (co[band, 1].min() + co[band, 1].max()))


def rebuild_empties(obj):
    """(Re)create one Image Empty per active reference, scaled so the figure height equals the model height, the feet on the
    ground and the figure axis on the model axis.  Visibility follows the 'visible' flag."""
    H, z0 = body_height(obj)
    py = _plumb_y(obj)
    keep = set()
    for r in active_refs(obj):
        keep.add("CS_Ref_%d" % r["id"])
        e = bpy.data.objects.get("CS_Ref_%d" % r["id"])
        g = r.get("geom")
        if g is None:
            continue
        path = os.path.join(ROOT, r["file"])
        if e is None:
            img = bpy.data.images.load(path, check_existing=True)
            e = bpy.data.objects.new("CS_Ref_%d" % r["id"], None)
            e.empty_display_type = 'IMAGE'
            e.data = img
            bpy.context.scene.collection.objects.link(e)
        e.hide_render = True
        e.hide_select = True
        e.show_in_front = False
        e.use_empty_image_alpha = True
        e.color = (1, 1, 1, 0.42)
        e.show_empty_image_only_axis_aligned = True
        mpp = H / g["H_px"]
        e.empty_display_size = max(g["w"], g["h"]) * mpp
        e.empty_image_offset = (-g["axis"] / g["w"], -(g["h"] - g["bot"]) / g["h"])
        v = r["view"]
        flip = -1.0 if (v == "side" and r.get("facing") == "L") else 1.0
        e.scale = (flip, 1.0, 1.0)
        if v == "side":
            e.rotation_euler = (math.radians(90), 0, math.radians(-90))
            e.location = (PLANE, py, z0)
        elif v == "back":
            e.rotation_euler = (math.radians(90), 0, math.radians(180))
            e.location = (0.0, -PLANE, z0)
        else:
            e.rotation_euler = (math.radians(90), 0, 0)
            e.location = (0.0, PLANE, z0)
        e.hide_viewport = not r.get("visible", True)
    for o in list(bpy.data.objects):
        if o.name.startswith("CS_Ref_") and o.name not in keep:
            bpy.data.objects.remove(o, do_unlink=True)


def analyse_references(obj, tag="refs"):
    """Run the silhouette analysis on all active references -> reference-metric dict (also stored on the object)."""
    act = active_refs(obj)
    if not act:
        raise RuntimeError("没有参考图")
    items = [dict(path=os.path.join(ROOT, r["file"]), view=r["view"] if r["view"] in ("front", "side", "back") else "front",
                  facing=r.get("facing")) for r in act]
    out = _analysis_path(tag)
    res = _run_analyzer(items, out)
    res["analysis_file"] = out
    obj["cs_ref_metrics"] = json.dumps(res, ensure_ascii=False)
    return res


def ref_override(res):
    return dict(profile=res["profile"], legs=res["legs"], landmarks=res.get("landmarks", {}))


# =========================================================== fit generator (reference -> draft)
NEUTRAL_FIXED = {"bust_pos": 0.0, "bust_dist": 0.0, "bust_vol_v": 0.0}


class FitJob:
    """Coordinate-search fit of the existing controls to a reference-metric dict, advanced in small slices so the UI stays alive."""

    def __init__(self, obj, res, init=None, fixed=None, steps=(0.4, 0.2, 0.1, 0.05), macro_steps=(0.15, 0.07, 0.03), sweeps=3):
        self.obj, self.res = obj, res
        self.fixed = dict(NEUTRAL_FIXED if fixed is None else fixed)
        C.REF_OVERRIDE = ref_override(res)
        A.P.off = {}
        A.save_props(obj)
        self.F = F.Fitter(obj, C.load_zones(obj))
        self.F.preload()
        st = C.default_state()
        if init:
            st.update(init)
        st.update(self.fixed)
        self.state = st
        skip = ("firmness", "head_scale_h", "uarm_len", "larm_len", "uarm_c") + tuple(self.fixed)
        self.free = [c["id"] for c in C.CONTROLS if c["stage"] in ("①", "①试") and c["id"] not in skip]
        self.steps, self.macro_steps, self.sweeps = steps, macro_steps, sweeps
        self.done = False
        self.progress = 0.0
        self.best = None
        self.obj0 = None
        self.gen = self._run()

    def _run(self):
        Fi = self.F
        state = self.state
        self.best = Fi.objective(state)
        self.obj0 = self.best
        nstep = len(self.steps) * self.sweeps
        k = 0
        for si, st in enumerate(self.steps):
            for sw in range(self.sweeps):
                improved = False
                for cid in self.free:
                    c = C.CTRL[cid]
                    s = self.macro_steps[min(si, len(self.macro_steps) - 1)] if c["kind"] == "macro" else st
                    cur = state[cid]
                    bestv, bestobj = cur, self.best
                    for d in (+s, -s):
                        lo, hi = F.bounds(c)
                        v = min(max(cur + d, lo), hi)
                        if abs(v - cur) < 1e-9:
                            continue
                        trial = dict(state)
                        trial[cid] = v
                        o = Fi.objective(trial)
                        if o < bestobj - 1e-6:
                            bestobj, bestv = o, v
                    if bestv != cur:
                        state[cid] = bestv
                        self.best = bestobj
                        improved = True
                    yield
                k += 1
                self.progress = k / nstep
                if not improved:
                    k += self.sweeps - sw - 1
                    break
        self.done = True
        self.progress = 1.0

    def advance(self, budget=0.25):
        t = time.time()
        while not self.done and time.time() - t < budget:
            try:
                next(self.gen)
            except StopIteration:
                self.done = True
        return self.done

    def run_all(self):
        while not self.advance(5.0):
            pass
        return self.finish()

    def finish(self):
        mp, lm = self.F.measure(self.state)
        rows = self.F.rows(mp, lm)
        full = C.apply_state(self.obj, self.state)
        return full, rows, mp


def fit_rows_report(rows):
    out = []
    for fl, q, w, why, ref, val in rows:
        out.append(dict(fl=fl, q=q, why=why, ref=round(float(ref), 4), got=round(float(val), 4), err=round(float(val - ref), 4)))
    return out


# =========================================================== prompt understanding (规则指令模式)
MODE_LABEL = "规则指令模式（词表规则，不是大模型；只认下表里的词，认不出的会原样列出）"
# (regex, label, {control: target})   target = absolute control value (bi: -0.8..0.8, macro: 0..1)
VOCAB = [
    (r"娇小|个子小|矮个|小个子", "整体身高：偏矮", {"height": 0.36}),
    (r"高挑|高个|个子高|身材高", "整体身高：偏高", {"height": 0.64}),
    (r"中等身高|中等个", "整体身高：中等", {"height": 0.5}),
    (r"纤细|苗条|骨感|偏瘦|很瘦|瘦", "体重：偏瘦", {"weight": 0.34}),
    (r"微胖|丰腴|肉感", "体重：偏丰腴", {"weight": 0.62}),
    (r"偏胖|很胖|胖", "体重：偏胖", {"weight": 0.70}),
    (r"匀称|标准身材|标准体型", "体重：匀称", {"weight": 0.5}),
    (r"健美|肌肉|运动型|健身|有力", "肌肉：偏强", {"muscle": 0.72, "shoulder_w": 0.3, "vshape": 0.3}),
    (r"柔软|纤弱", "肌肉：偏弱", {"muscle": 0.32}),
    (r"巨乳|大胸|胸大|丰满的胸|丰胸|胸部丰满|胸部很大|胸很大", "胸：大", {"cupsize": 1.0, "bust_c": 0.5, "bust_round": 0.7, "bust_upper": 0.5}),
    (r"中等胸|胸部适中|胸适中", "胸：中等", {"cupsize": 0.62, "bust_c": 0.1}),
    (r"小胸|平胸|贫乳|胸小|胸部小", "胸：小", {"cupsize": 0.3, "bust_c": -0.5}),
    (r"胸挺|高耸|胸位高|上提", "胸位：高", {"bust_pos": 0.5}),
    (r"乳沟|胸聚拢|胸靠拢", "胸间距：靠拢", {"bust_dist": -0.6}),
    (r"细腰|小蛮腰|腰细|腰很细", "腰：细", {"waist_c": -0.65, "underbust_c": -0.3}),
    (r"粗腰|水桶腰|腰粗", "腰：粗", {"waist_c": 0.5}),
    (r"小腹|肚子", "小腹：有", {"belly": 0.5}),
    (r"宽胯|大胯|胯宽|胯部宽", "胯：宽", {"hip_w": 0.55, "hip_c": 0.5}),
    (r"窄胯|胯窄|胯小", "胯：窄", {"hip_w": -0.5, "hip_c": -0.4}),
    (r"翘臀|大臀|丰臀|蜜桃臀|臀大|臀部丰满|臀部很翘", "臀：大且翘", {"glute_vol": 0.4, "glute_proj": 0.8, "hip_c": 0.3, "glute_low": 0.8}),
    (r"小臀|平臀|臀小|臀部小", "臀：小", {"glute_vol": -0.5, "glute_proj": 0.0}),
    (r"长腿|大长腿|腿长|腿很长", "腿：长", {"thigh_len": 0.5, "shin_len": 0.5}),
    (r"短腿|腿短", "腿：短", {"thigh_len": -0.5, "shin_len": -0.5}),
    (r"细腿|腿细|腿很细", "腿：细", {"thigh_c": -0.4, "thigh_fat": -0.4, "calf_c": -0.3}),
    (r"粗腿|腿粗|肉腿|大腿丰满", "腿：粗", {"thigh_c": 0.5, "thigh_fat": 0.5, "calf_c": 0.2}),
    (r"并腿|腿并拢|X型腿", "站姿：并腿", {"legs_together": 0.9, "knees_in": 0.6}),
    (r"宽肩|肩宽|肩膀宽", "肩：宽", {"shoulder_w": 0.6}),
    (r"窄肩|削肩|肩窄|肩膀窄", "肩：窄", {"shoulder_w": -0.5}),
    (r"沙漏|S型|s型|S形", "体型：沙漏", {"cupsize": 0.9, "waist_c": -0.6, "underbust_c": -0.3, "hip_w": 0.4, "hip_c": 0.5, "glute_vol": 0.3, "glute_proj": 0.5}),
    (r"梨形|梨型|下重上轻", "体型：梨形", {"hip_c": 0.6, "hip_w": 0.5, "thigh_c": 0.4, "cupsize": 0.45, "shoulder_w": -0.3, "waist_c": -0.2, "glute_vol": 0.3}),
    (r"苹果型|苹果形", "体型：苹果", {"belly": 0.5, "waist_c": 0.3, "hip_c": -0.1, "thigh_c": -0.2}),
    (r"倒三角", "体型：倒三角", {"shoulder_w": 0.5, "vshape": 0.6, "hip_w": -0.3, "hip_c": -0.2}),
]
LATER_VOCAB = [
    (r"脸|五官|眼|眉|鼻|嘴|唇|牙|耳|发型|头发|刘海|发色|长发|短发", "头脸/头发：后续阶段（阶段 C/D），本阶段只记录不执行"),
    (r"皮肤|肤色|材质|纹理|指甲|瞳|妆", "皮肤/材质：后续阶段，只记录"),
    (r"手指|脚趾|手型|脚型|乳头|乳晕", "手足/胸细节：后续阶段，只记录"),
    (r"年龄|岁|少女|熟女|萝莉", "年龄/气质：本阶段无对应控制（只有体型），只记录"),
    (r"衣|裙|裤|鞋|内衣|泳装|服装", "服装：不在本系统范围，只记录"),
]
MOD_UP = re.compile(r"非常|特别|极其|极|超|很")
MOD_DN = re.compile(r"稍微|略微|略|稍|有点|一点|微")


def understand(text):
    """-> dict(mode, text, recognized[{phrase,label,targets}], unrecognized, later, conflicts, height_m, state_targets)."""
    out = dict(mode=MODE_LABEL, text=text, recognized=[], unrecognized=[], later=[], conflicts=[], notes=[], state_targets={})
    parts = [p.strip() for p in re.split(r"[，,；;。\n、]|并且|而且|同时|还有|以及|和|，", text) if p.strip()]
    for part in parts:
        hit = False
        m = re.search(r"(1[3-9]\d|2[0-1]\d)\s*(?:cm|厘米|CM)", part)
        if m:
            hm = int(m.group(1)) / 100.0
            hv = min(max(0.5 + (hm - HEIGHT_M_AT_HALF) / HEIGHT_M_PER_UNIT, C.CTRL["height"]["lo"]), C.CTRL["height"]["hi"])
            out["recognized"].append(dict(phrase=part, label="身高 %.2f m（宏参数换算，MPFB 的米数是估算，误差约 ±3 cm）" % hm, targets={"height": round(hv, 3)}))
            out["state_targets"]["height"] = round(hv, 3)
            out["height_m"] = hm
            hit = True
        m = re.search(r"(\d{2,3})\s*(?:kg|公斤|千克|KG)", part)
        if m:
            kg = int(m.group(1))
            wv = 0.35 if kg <= 48 else (0.5 if kg <= 62 else (0.62 if kg <= 75 else 0.7))
            out["recognized"].append(dict(phrase=part, label="体重 %d kg → 体重宏 %.2f（只能粗略映射：同样体重因身高不同形态不同）" % (kg, wv), targets={"weight": wv}))
            out["state_targets"]["weight"] = wv
            out["notes"].append("体重数值只能粗略映射为偏瘦/匀称/偏胖")
            hit = True
        covered = []
        for rx, label, tg in VOCAB:
            mm = re.search(rx, part)
            if not mm or any(a <= mm.start() and mm.end() <= b for a, b in covered):
                continue
            covered.append((mm.start(), mm.end()))
            hit = True
            k = 1.0
            pre = part[max(0, mm.start() - 4):mm.start()]
            if MOD_UP.search(pre):
                k = 1.25
            elif MOD_DN.search(pre):
                k = 0.55
            t = {}
            for cid, v in tg.items():
                c = C.CTRL[cid]
                neutral = c["default"]
                t[cid] = round(min(max(neutral + (v - neutral) * k, c["lo"]), c["hi"]), 3)
            for cid, v in t.items():
                old = out["state_targets"].get(cid)
                if old is not None and abs(old - v) > 0.15 and (old - C.CTRL[cid]["default"]) * (v - C.CTRL[cid]["default"]) < 0:
                    out["conflicts"].append("「%s」与前面的描述对「%s」方向相反（%.2f vs %.2f），已采用后出现的" % (part, C.CTRL[cid]["label"], old, v))
                out["state_targets"][cid] = v
            out["recognized"].append(dict(phrase=part, label=label + ("（程度词：×%.2f）" % k if k != 1.0 else ""), targets=t))
        for rx, why in LATER_VOCAB:
            if re.search(rx, part):
                out["later"].append(dict(phrase=part, reason=why))
                hit = True
                break
        if not hit:
            out["unrecognized"].append(part)
    return out


def state_from_understanding(und, base=None):
    st = dict(base or C.default_state())
    st.update({k: v for k, v in und["state_targets"].items() if k in st})
    return C.clamp_state(st)


# =========================================================== presets (no-image creation)
def _p(**kw):
    return kw


BUILTIN = [
    ("标准均衡", "默认比例的成年女性，腰略收", _p(cupsize=0.6, waist_c=-0.2, hip_c=0.15, weight=0.5)),
    ("沙漏丰满", "胸臀丰满、腰细", _p(height=0.55, cupsize=0.9, bust_c=0.4, bust_round=0.5, bust_pos=0.3, waist_c=-0.6, underbust_c=-0.3, hip_w=0.4,
                                     hip_c=0.5, glute_vol=0.3, glute_proj=0.5, thigh_c=0.3)),
    ("梨形", "下身偏重、肩窄、胸中等", _p(cupsize=0.45, shoulder_w=-0.3, waist_c=-0.2, hip_w=0.5, hip_c=0.6, glute_vol=0.4, glute_proj=0.4, thigh_c=0.4, thigh_fat=0.3, weight=0.55)),
    ("高挑苗条", "高、瘦、长腿、胸臀小", _p(height=0.66, weight=0.35, cupsize=0.4, waist_c=-0.4, hip_c=-0.1, thigh_len=0.5, shin_len=0.4, thigh_c=-0.4,
                                        thigh_fat=-0.4, calf_c=-0.3, muscle=0.35, bust_c=-0.2)),
    ("娇小", "矮个、比例紧凑", _p(height=0.36, weight=0.45, cupsize=0.5, waist_c=-0.2, thigh_len=-0.2, shin_len=-0.1)),
    ("健美", "肌肉感、肩宽、腰收", _p(muscle=0.72, shoulder_w=0.4, chest_w=0.3, vshape=0.4, waist_c=-0.4, glute_vol=0.3, thigh_c=0.3, cupsize=0.55, weight=0.5)),
    ("丰腴曲线", "体重偏高、胸臀大、腰不细", _p(weight=0.66, cupsize=0.85, belly=0.4, waist_c=0.1, hip_w=0.4, hip_c=0.6, glute_vol=0.5, thigh_fat=0.5, thigh_c=0.4)),
    ("少女纤细", "窄肩、小胸、纤细", _p(weight=0.32, cupsize=0.35, bust_c=-0.3, waist_c=-0.3, hip_c=-0.3, shoulder_w=-0.3, thigh_c=-0.5, calf_c=-0.3)),
]


def builtin_presets():
    """-> list of (name, description, full state dict)."""
    out = []
    for name, desc, kv in BUILTIN:
        st = C.default_state()
        st.update(kv)
        out.append((name, desc, C.clamp_state(st)))
    p5 = os.path.join(C.VERS, "s1_draft_v005_state.json")
    if os.path.isfile(p5):
        st = C.default_state()
        st.update(json.load(open(p5, encoding="utf-8"))["state"])
        out.append(("参考图ref01拟合(v005)", "v005 交付草模的参数（对参考图 ref01 的拟合结果）", C.clamp_state(st)))
    return out


def user_presets():
    os.makedirs(USER_PRESETS, exist_ok=True)
    res = []
    for f in sorted(os.listdir(USER_PRESETS)):
        if f.endswith(".json"):
            d = json.load(open(os.path.join(USER_PRESETS, f), encoding="utf-8"))
            res.append((d["name"], d.get("note", ""), d))
    return res


def save_user_preset(obj, name, note=""):
    """Saves the current state + point offsets + mirror under a name (never overwrites: a taken name gets a numeric suffix)."""
    os.makedirs(USER_PRESETS, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|\s]+', "_", name.strip()) or "preset"
    path = os.path.join(USER_PRESETS, safe + ".json")
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(USER_PRESETS, "%s_%d.json" % (safe, n))
    d = dict(name=name.strip() or safe, note=note, saved=time.strftime("%Y-%m-%d %H:%M:%S"), state=C.get_state(obj),
             offsets={k: v for k, v in A.P.off.items()}, mirror=A.P.mirror, source_blend=os.path.basename(bpy.data.filepath))
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return path


def apply_preset_data(obj, state, offsets=None, mirror=True):
    """Put a preset on the body (offsets replaced). Returns the full state."""
    A.P.off = {k: list(v) for k, v in (offsets or {}).items()}
    A.P.mirror = bool(mirror)
    A.save_props(obj)
    return C.apply_state(obj, C.clamp_state(dict(state)))


# =========================================================== free-form AI entry
AI_STATUS = "未完成：本机没有被授权的 AI 环境（不接入 API key / 付费服务 / MCP）。这里只能把请求写进收件箱，由 Claude Code 会话人工处理，不是面板内的 AI。"


def submit_free_ai(text):
    os.makedirs(os.path.dirname(INBOX), exist_ok=True)
    new = not os.path.exists(INBOX)
    with open(INBOX, "a", encoding="utf-8") as f:
        if new:
            f.write("# AI 收件箱（自由 AI 入口：未完成）\n\n面板内没有授权的 AI 环境，下面的请求只被记录，需在 Claude Code 会话中人工处理。\n\n")
        f.write("- %s  「%s」 → 状态：未处理（人工）\n" % (time.strftime("%Y-%m-%d %H:%M"), text))
    return AI_STATUS
