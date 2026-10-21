"""v3 body-type inputs (A3 体型入口): runtime overlay on a_inputs (a_inputs.py unchanged).

* 胖瘦 is ONE number, -0.5 .. 1 (v3_body.fat_to_state): 0..1 = normal range, -0.5 = extreme thin (real deformation with
  MPFB's native local fat-decrease targets, never a negative weight macro, never silently clamped: the effective value is shown).
* muscle and shoulder width are separate entries ("健美" no longer widens the shoulders).
* prompt words for thin / fat go through 胖瘦; kg -> 胖瘦 is a coarse table (said so in the result).
* 5 body-type presets change ONLY the fat / muscle fields (胖瘦, muscle, belly, thigh / upper-arm / calf fat); proportions,
  bust / glute design, points and parts are kept.
* user presets also store the parts / interface data when the parts system is present.
* honest labels: the prompt entry is a word-table RULE system, not an AI model."""
import re
import json
import numpy as np

import s1_core as C
import a_inputs as I
import v3_body as VB

FAT_NEUTRAL = 0.40              # 胖瘦 of the "standard" body (v020 weight 0.43)
# prompt words -> 胖瘦 (absolute).  Encoded as v3_fatdist = 胖瘦 - FAT_NEUTRAL so a_inputs' degree words (很/稍微 ...) scale
# around the standard body; converted back in understand().
FAT_WORDS = [
    (r"骨感|极瘦|皮包骨|骨瘦如柴|消瘦", "胖瘦：极瘦（扩展范围）", -0.40),
    (r"纤细|苗条|偏瘦|很瘦|瘦", "胖瘦：偏瘦", 0.10),
    (r"匀称|标准身材|标准体型", "胖瘦：匀称", 0.40),
    (r"微胖|丰腴|肉感", "胖瘦：微胖", 0.65),
    (r"肥胖|很胖|偏胖|胖", "胖瘦：偏胖/肥胖", 0.90),
]
KG_TABLE = ((45, 0.0), (52, 0.2), (60, 0.4), (68, 0.6), (78, 0.8), (999, 1.0))

# fields a body-type preset may change (everything else - proportions, bust / glute design, points, parts - is kept)
TYPE_FIELDS = ("muscle", "belly", "thigh_fat", "uarm_fat", "calf_fat")
# values chosen by health test (v3_exp_types*.py): MPFB's muscle macro below ~0.5 flips faces at the eyelid and above ~0.65 creases
# the bust midline, so muscle stays 0.50..0.62; 匀称 = the confirmed stage-A body's own values
BODY_TYPES = [
    ("柔和纤瘦", "瘦，四肢和小腹脂肪少，肌肉标准", dict(fat=0.10, muscle=0.50, belly=-0.3, thigh_fat=-0.55, uarm_fat=-0.2, calf_fat=-0.3)),
    ("精瘦有肌肉", "脂肪少、肌肉较明显（肩宽不变，肩宽单独调）", dict(fat=0.20, muscle=0.62, belly=-0.3, thigh_fat=-0.55, uarm_fat=-0.2, calf_fat=-0.25)),
    ("匀称", "已确认的阶段A身体（胖瘦0.40）", dict(fat=0.40, muscle=0.537, belly=-0.221, thigh_fat=-0.55, uarm_fat=0.0, calf_fat=-0.25)),
    ("丰满", "整体偏丰满，大腿、上臂有脂肪", dict(fat=0.70, muscle=0.537, belly=0.0, thigh_fat=-0.2, uarm_fat=0.2, calf_fat=-0.1)),
    ("肥胖", "胖瘦 1.0，腰腹、大腿、上臂脂肪多", dict(fat=1.00, muscle=0.50, belly=0.0, thigh_fat=0.0, uarm_fat=0.3, calf_fat=0.1)),
]
MODE_LABEL_V3 = ("规则指令模式（词表规则，不是大模型；只认词表里的词，认不出的原样列出）。"
                 "“胖瘦”类词统一换算成胖瘦数值 -0.5～1，体重kg 只是粗略对照表")

_ORIG = {}


def body_type_state(st, name):
    """state with body type `name` applied (only 胖瘦 + TYPE_FIELDS change)"""
    d = dict(next(v for n, _, v in BODY_TYPES if n == name))
    out = dict(st)
    out.update(VB.fat_to_state(d.pop("fat")))
    out.update(d)
    return out


def set_fat(st, f):
    """state with 胖瘦 = f (raises ValueError outside -0.5..1: the UI shows the message, nothing is clamped silently)"""
    out = dict(st)
    out.update(VB.fat_to_state(float(f)))
    return out


def kg_to_fat(kg):
    for lim, f in KG_TABLE:
        if kg <= lim:
            return f
    return 1.0


NEG = r"(?:不要|不想要|不想|不需要|不用|无需|别太|别|不太|不)"
SPLIT = r"但是|但|不过|可是"


def _negations(text):
    """'不要宽肩' style phrases: the vocabulary word is negated -> its fields stay at the standard value.  Returns (text with the
    negated phrases cut out, recognized items)"""
    items = []
    for rx, label, tg in list(I.VOCAB):
        for mm in list(re.finditer(NEG + "(?:" + rx + ")", text)):
            items.append(dict(phrase=mm.group(0), label="%s → 否定（“%s”）：保持标准值，不改" % (label.split("（")[0], mm.group(0)),
                              targets={k: C.CTRL[k]["default"] for k in tg}))
            text = text.replace(mm.group(0), "，", 1)
    return text, items


def understand(text):
    t0 = text
    text, neg = _negations(re.sub(SPLIT, "，", text))
    und = _ORIG["understand"](text)
    und["text"] = t0
    und["mode"] = MODE_LABEL_V3
    for r in neg:
        und["recognized"].append(r)
        und["state_targets"].update(r["targets"])
    f = None
    for r in und["recognized"]:
        t = r["targets"]
        if "v3_fatdist" in t:
            fr = FAT_NEUTRAL + t.pop("v3_fatdist")
            fe = float(np.clip(fr, VB.FAT_LO, VB.FAT_HI))
            r["label"] += "  → 胖瘦 %.2f" % fe + ("（程度词后超出 -0.5～1，已按边界 %.2f 采用）" % fe if abs(fe - fr) > 1e-6 else "")
            f = fe
        if "weight" in t and r["label"].startswith("体重"):
            m = re.search(r"(\d{2,3})\s*(?:kg|公斤|千克|KG)", r["phrase"])
            if m:
                f = kg_to_fat(int(m.group(1)))
                t.pop("weight")
                r["label"] = "体重 %s kg → 胖瘦 %.2f（粗略对照表：同样体重因身高不同形态不同）" % (m.group(1), f)
        if f is not None:
            t.update({k: round(v, 3) for k, v in VB.fat_to_state(f).items()})
    st_t = und["state_targets"]
    st_t.pop("v3_fatdist", None)
    if f is not None:
        st_t.update({k: round(v, 3) for k, v in VB.fat_to_state(f).items()})
        und["fat"] = f
        und["notes"].append("胖瘦 %.2f：%s" % (f, VB.fat_effective_text(dict(st_t))))
    return und


def _parts_mod():
    try:
        import v3_parts
        return v3_parts
    except ImportError:
        return None


def save_user_preset(obj, name, note=""):
    path = _ORIG["save_user_preset"](obj, name, note)
    PM = _parts_mod()
    if PM is not None and PM.has_parts(obj):
        d = json.load(open(path, encoding="utf-8"))
        d["parts"] = PM.snapshot(obj)
        d["fat"] = round(VB.state_to_fat(d["state"]), 4)
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return path


def apply_preset_data(obj, state, offsets=None, mirror=True, parts=None):
    """parts data (if given) is put in place BEFORE the body is rebuilt, so the preset's part edits are in the result.  Unsaved
    hand edits on the part objects are dropped (a preset replaces the whole design; undo brings them back)."""
    PM = _parts_mod()
    if PM is not None and PM.has_parts(obj):
        PM.write_parts(obj)
        if parts:
            PM.restore(obj, parts)
    return _ORIG["apply_preset_data"](obj, state, offsets, mirror)


def register():
    if _ORIG:
        return
    _ORIG.update(understand=I.understand, save_user_preset=I.save_user_preset, apply_preset_data=I.apply_preset_data)
    old = [v for v in I.VOCAB if not any(k in v[2] for k in ("weight",))]
    for i, (rx, label, tg) in enumerate(old):
        if label.startswith("肌肉：偏强"):
            old[i] = (rx, "肌肉：偏强（肩宽不变，肩宽单独说“宽肩”）", {"muscle": 0.72})
    I.VOCAB[:] = [(rx, lab, {"v3_fatdist": round(f - FAT_NEUTRAL, 3)}) for rx, lab, f in FAT_WORDS] + old
    I.MODE_LABEL = MODE_LABEL_V3
    I.understand = understand
    I.save_user_preset = save_user_preset
    I.apply_preset_data = apply_preset_data
