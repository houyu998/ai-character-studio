"""Stage-1 panel for Blender's 3D-viewport sidebar (N panel -> tab "角色工作室").

Start with  启动阶段1草模.bat  (opens versions/s1_draft_v001.blend and runs s1_panel_boot.py).

What it gives the user (stage ① only: overall proportions + chest/hip big form):
  * bounded sliders for every stage-① control (MPFB "已有" ones and the trial "新增" ones, labelled)
  * views (front / side / back / 3-4), clay / zone-colour / material shading
  * undo / redo (own history, drag steps are merged), before/after toggle with three kinds of "before"
  * natural-language edits: parsed into an explicit operation (部位 -> 属性 -> 方向/程度 -> 保护范围), really executed on the
    sliders, then the measured change (target and protected regions) is reported.  Anything that cannot be mapped
    is NOT executed and is written to docs/S1_todo_log.md instead
  * save as a NEW version (never overwrites), measure against the reference, before/after PNG sheet,
    "确认阶段①" (only the user presses it)
"""
import os
import re
import sys
import glob
import json
import time
import math
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from bpy.props import FloatProperty, StringProperty, EnumProperty, PointerProperty

import s1_core as C
import s1_custom as U
import a_points as A

BODY = "CS_Body"
TAB = "Item"      # the sidebar tab that is open by default (a script cannot switch tabs; a custom tab would need a click)
TODO_LOG = os.path.join(C.ROOT, "docs", "S1_todo_log.md")
STAGE_CTRL = ("①", "①试", "B")      # stage-A controls stay available in stage B (confirmed big proportions are protected by the before/after check)

S = types.SimpleNamespace(hist=[], hi=-1, before=None, before_name="草模", after=None, showing_before=False,
                          log=[], sync=False, prof=None, before_pts=None, after_pts=None, health=None, health_prev=None,
                          health_on=True, tool_on=False, ref_sel=-1)


def stage_of(obj=None):
    """"A" or "B": which stage the open file is in (object property cs_stage, default A)"""
    obj = obj if obj is not None else body()
    return "B" if (obj is not None and str(obj.get("cs_stage", "")).startswith("B")) else "A"


STAGE_INFO = {
    "A": dict(title="阶段 A 体型构建（待你确认）", name="阶段 A 体型构建", prefix="s1_confirmed",
              items=["① 整体高矮胖瘦", "② 肩腰胯与腿长比例", "③ 胸臀的大体大小和位置"]),
    "B": dict(title="阶段 B 身体塑形（待你确认）", name="阶段 B 身体塑形", prefix="s1_confirmedB",
              items=["① 胸部造型", "② 腰臀四肢造型", "③ 手脚造型"]),
}


def body():
    o = bpy.data.objects.get(BODY)
    if o is not None:
        return o
    try:
        return C.find_basemesh()
    except RuntimeError:
        return None


def log(*lines):
    for ln in lines:
        S.log.append(ln)
    del S.log[:-14]


# ------------------------------------------------------------------ state <-> properties
def exposed():
    return [c for c in C.CONTROLS if c["stage"] in STAGE_CTRL]


def props_to_state(scene):
    st = C.get_state(body())
    for c in exposed():
        st[c["id"]] = float(getattr(scene.cs, "c_" + c["id"]))
    return st


def sync_props(scene, state):
    S.sync = True
    try:
        for c in exposed():
            setattr(scene.cs, "c_" + c["id"], float(state.get(c["id"], c["default"])))
    finally:
        S.sync = False


def commit(state, label, coalesce=None):
    """Apply a state to the mesh and push it on the history (drag steps of one slider are merged)."""
    obj = body()
    try:
        full = C.apply_state(obj, state)
    except ValueError as ex:                      # non-finite value: nothing was applied, nothing is recorded
        log("✗ 已拒绝并回滚：%s" % ex)
        return C.get_state(obj)
    now = time.time()
    top = S.hist[S.hi] if 0 <= S.hi < len(S.hist) else None
    A.save_props(obj)
    if coalesce and top is not None and top.get("co") == coalesce and S.hi == len(S.hist) - 1 and now - top["t"] < 1.5:
        top.update(state=full, t=now, pts=A.snapshot())
    else:
        del S.hist[S.hi + 1:]
        S.hist.append(dict(label=label, state=full, t=now, co=coalesce, pts=A.snapshot()))
        S.hi = len(S.hist) - 1
    S.after = full
    S.after_pts = A.snapshot()
    _health_after_commit()
    return full


def _health_after_commit(report=True):
    """run the mesh health check after every committed change; remember the previous result to tell NEW problems from old ones.
    report=False (undo / redo): the check is refreshed but no 'new problem' message is logged - going back to an earlier state is not a new edit"""
    if not S.health_on:
        return
    try:
        import a_health as H
        S.health_prev = S.health
        S.health = H.check(body())
        new = H.new_since(S.health_prev, S.health) if (report and S.health_prev is not None) else []
        if new:
            log("⚠ 这一步新增 %d 处网格问题（%s）——见面板「网格健康检查」，可定位，可点「撤销这一步」" % (
                len(new), "、".join("%s@%s %.0f%s" % (i["kind"], i["region"], i["value"], i["unit"]) for i in new[:3])))
    except Exception as ex:                       # a failing check must never break editing
        S.health = None
        log("健康检查出错：%s" % ex)


def _updater(cid):
    def _u(self, context):
        if S.sync or S.showing_before or body() is None:
            return
        commit(props_to_state(context.scene), "拖动：" + C.CTRL[cid]["label"], coalesce=cid)
    return _u


def _shade_update(self, context):
    apply_shading(context.screen, self.shade)


def apply_shading(screen, mode):
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        sh = area.spaces.active.shading
        sh.type = 'SOLID'
        sh.light = 'STUDIO'
        try:
            sh.studio_light = 'basic.sl'        # left/right symmetric lighting
        except (TypeError, ValueError):
            pass
        sh.show_cavity = True
        if mode == "ZONE":
            sh.color_type = 'VERTEX'
            o = body()
            if o is not None:
                ca = o.data.color_attributes.get("zone_color")
                if ca is not None:
                    o.data.color_attributes.active_color = ca
        elif mode == "CLAY":
            sh.color_type = 'SINGLE'
            sh.single_color = (0.66, 0.66, 0.66)
        else:
            sh.color_type = 'MATERIAL'


def make_props():
    ann = {}
    for c in exposed():
        ann["c_" + c["id"]] = FloatProperty(name=c["label"], min=c["lo"], max=c["hi"], default=c["default"],
                                            precision=2, update=_updater(c["id"]), description=c.get("note", ""))
    ann["nl_text"] = StringProperty(name="", default="", description="例如：胯部加宽一点，腰再细一点，臀更翘")
    ann["shade"] = EnumProperty(name="显示", items=[("CLAY", "灰模", ""), ("ZONE", "分区着色", ""), ("MAT", "材质", "")],
                                default="CLAY", update=_shade_update)
    return type("CSProps", (bpy.types.PropertyGroup,), {"__annotations__": ann})


# ------------------------------------------------------------------ measurement
HEIGHT_M_PER_UNIT = 1.33      # measured with MPFB: height macro 0.5 -> 1.496 m, 0.65 -> 1.696 m (this body mix)
MET = [("身高", None, None), ("胸前凸", 0.725, "side_fwd"), ("肩/胸廓宽", 0.725, "front_w"), ("腰宽(最细)", 0.65, "front_w"),
       ("胯宽", 0.525, "front_w"), ("臀后凸", 0.525, "side_back"), ("大腿单腿宽", 0.40, "leg_w"),
       ("小腿单腿宽", 0.25, "leg_w")]
MET_NAMES = [m[0] for m in MET]


def measure():
    obj = body()
    if S.prof is None or S.prof[0] != obj.name:
        S.prof = (obj.name, C.Profiler(obj, C.load_zones(obj), fls=[0.25, 0.4, 0.525, 0.65, 0.725]))
    mp = S.prof[1](C.raw_coords(obj))
    out = {}
    for name, fl, q in MET:
        out[name] = mp["H"] if fl is None else mp["profile"][fl][q]
    return out


def fmt_met(name, v):
    return "%.2f m" % v if name == "身高" else "%.1f%%H" % (v * 100)


def ref_met():
    out = {}
    for name, fl, q in MET:
        if fl is not None:
            out[name] = C.ref_value(fl, q)
    return out


# ------------------------------------------------------------------ natural-language edits
# rule = (part/attribute regex, [control ids], label, affects(metrics), default direction or None)
# dflt: None = direction must be in the sentence; +1/-1 = used when the sentence gives no direction word;
#       ("F", +1/-1) = the part word itself fixes the direction (e.g. 靠拢, 内收, 下移); only 减/取消/不要 flips it.
RULES = [
    # ---- stage B (body shaping): bust / waist / hip / limbs / hands / feet detail
    (r"胸.{0,3}(下垂|垂)|胸.{0,2}(下坠)", ["bust_sag"], "胸 上托↔下垂", ["胸前凸"], ("F", +1)),
    (r"胸.{0,3}(上托|托高|托起)", ["bust_sag"], "胸 上托↔下垂", ["胸前凸"], ("F", -1)),
    (r"胸.{0,3}(外展|朝外|外扩|向外)", ["bust_yaw"], "胸 朝向（内收↔外展）", ["胸前凸"], ("F", +1)),
    (r"胸.{0,3}(内收|朝内|向内)", ["bust_yaw"], "胸 朝向（内收↔外展）", ["胸前凸"], ("F", -1)),
    (r"胸.{0,3}(朝上|上翘|上扬)", ["bust_pitch"], "胸 朝向（朝下↔朝上）", ["胸前凸"], ("F", +1)),
    (r"胸.{0,3}(朝下)", ["bust_pitch"], "胸 朝向（朝下↔朝上）", ["胸前凸"], ("F", -1)),
    (r"下半球|胸下部|下胸饱满|胸.{0,2}下.{0,2}饱满", ["bust_lowfull"], "胸 下半球饱满度", ["胸前凸"], None),
    (r"臀峰.{0,2}(上提|提高|抬高|上移|上)|臀.{0,2}(上提|提高|抬高)", ["glute_up"], "臀峰高度（下↔上）", ["臀后凸"], ("F", +1)),
    (r"臀峰.{0,2}(下移|降低|放低|下)", ["glute_up"], "臀峰高度（下↔上）", ["臀后凸"], ("F", -1)),
    (r"臀腿.{0,2}过渡|臀.{0,2}腿.{0,2}(过渡|衔接)|臀下.{0,2}(过渡|柔和)", ["glute_leg"], "臀↔腿过渡柔化", ["臀后凸"], ("F", +1)),
    (r"自动细化|AI细化|AI 细化|修尖点|修折痕|去尖|去折痕|平滑尖点", ["refine_auto"], "AI 自动细化（只修尖点/折痕）", [], ("F", +1)),
    (r"肚脐.{0,3}(上移|提高|抬高|往上|向上|上)", ["navel_ud"], "肚脐 上↔下", [], ("F", -1)),
    (r"肚脐.{0,3}(下移|降低|往下|向下|下)", ["navel_ud"], "肚脐 上↔下", [], ("F", +1)),
    (r"肚脐.{0,2}(凹|深)", ["navel_io"], "肚脐 凹↔凸", [], ("F", -1)),
    (r"肚脐.{0,2}(凸|浅)", ["navel_io"], "肚脐 凹↔凸", [], ("F", +1)),
    (r"腹.{0,2}(紧实|紧致|平坦)|小腹.{0,2}(紧实|紧致)", ["abd_tone"], "腹部 紧实↔柔软", [], ("F", -1)),
    (r"腹.{0,2}(柔软|松软|软)", ["abd_tone"], "腹部 紧实↔柔软", [], ("F", +1)),
    (r"三角肌|肩.{0,2}肌肉|肩部肌肉", ["shoulder_muscle"], "三角肌/肩部肌肉", [], None),
    (r"上臂.{0,2}肌肉|大臂.{0,2}肌肉", ["uarm_muscle"], "上臂肌肉", [], None),
    (r"上臂.{0,2}(脂肪|肉)|大臂.{0,2}(脂肪|肉)|拜拜肉", ["uarm_fat"], "上臂脂肪", [], None),
    (r"前臂.{0,2}肌肉|小臂.{0,2}肌肉", ["larm_muscle"], "前臂肌肉", [], None),
    (r"前臂.{0,2}(脂肪|肉)|小臂.{0,2}(脂肪|肉)", ["larm_fat"], "前臂脂肪", [], None),
    (r"大腿.{0,2}肌肉", ["thigh_muscle"], "大腿肌肉", [], None),
    (r"小腿.{0,2}肌肉|腿肚", ["calf_muscle"], "小腿肌肉", [], None),
    (r"膝围|膝盖.{0,2}(粗|细|大|小)", ["knee_c"], "膝围", [], None),
    (r"O型腿|o型腿", ["leg_valgus"], "腿型 O↔X", [], ("F", -1)),
    (r"腿型", ["leg_valgus"], "腿型 O↔X", [], None),
    (r"手指.{0,2}(长|短)|指长", ["finger_len"], "手指长度", [], None),
    (r"手指.{0,2}(粗|细)|指粗", ["finger_dia"], "手指粗细", [], None),
    (r"手指.{0,2}(张开|分开)", ["finger_dist"], "手指间距（并拢↔张开）", [], ("F", +1)),
    (r"手指.{0,2}(并拢|合拢|靠拢)", ["finger_dist"], "手指间距（并拢↔张开）", [], ("F", -1)),
    (r"腕围|手腕", ["wrist_c"], "腕围", [], None),
    (r"手.{0,2}(大小|整体)|手掌|手大|手小", ["hand_scale"], "手整体大小", [], None),
    (r"脚.{0,2}(长|短)|脚长", ["foot_len"], "脚长（前后）", [], None),
    (r"脚.{0,2}(宽|窄)|脚宽", ["foot_w"], "脚宽", [], None),
    (r"脚背|脚厚", ["foot_h"], "脚背高/脚厚", [], None),
    (r"踝围|脚踝|脚腕", ["ankle_c"], "踝围", [], None),
    (r"脚.{0,2}(大小|整体)|脚大|脚小", ["foot_scale"], "脚整体大小", [], None),
    # v002 additions (bust position / spacing / upper fullness moved into stage 1; three trial deformers)
    # v005: bust big-form trial deformers (roundness / upper fullness / lower-edge step) live in stage 1
    (r"胸.{0,3}(圆润|更圆|变圆|圆一点|丰润|隆起)|胸[形型].{0,3}圆|胸丘", ["bust_round"], "胸形圆润(新增·试验)", ["胸前凸"], ("F", +1)),
    (r"锁骨下|上胸.{0,3}(填充|填满)|胸上部.{0,3}填充", ["bust_upper"], "上胸饱满·锁骨下填充(新增·试验)", ["胸前凸"], ("F", +1)),
    (r"胸下缘|下胸缘|胸.{0,2}下缘|台阶|折棱|棱角", ["bust_lower"], "胸下缘圆润·抹平硬台阶(新增·试验)", ["胸前凸"], ("F", +1)),
    (r"乳沟|胸.{0,3}(靠拢|聚拢|并拢|靠近|挤)", ["bust_dist"], "胸间距(负=靠拢)", ["胸前凸"], ("F", -1)),
    (r"胸.{0,3}(拉开|分开|外扩)", ["bust_dist"], "胸间距(负=靠拢)", ["胸前凸"], ("F", +1)),
    (r"胸间距|胸距", ["bust_dist"], "胸间距(负=靠拢)", ["胸前凸"], None),
    (r"胸.{0,2}(上提|上移|抬高|提高|上托|挺拔)|上提|挺拔", ["bust_pos"], "胸根位置(升/降)", ["胸前凸"], ("F", +1)),
    (r"下垂|胸.{0,2}(下移|降低|放低)", ["bust_pos"], "胸根位置(升/降)", ["胸前凸"], ("F", -1)),
    (r"胸位置", ["bust_pos"], "胸根位置(升/降)", ["胸前凸"], None),
    (r"上胸|上半球|胸上部", ["bust_vol_v"], "胸上下体积分布(正=上半球更满)", ["胸前凸"], None),
    (r"膝.{0,2}(并拢|内收|靠拢|夹紧)|X型腿|x型腿", ["knees_in"], "膝并拢(新增·试验)", [], ("F", +1)),
    (r"膝.{0,2}(后移|反屈|后弓)|腿线.{0,2}后弓", ["knees_back"], "膝后移(新增·试验)", [], ("F", +1)),
    (r"下臀|臀下缘|臀峰下移|臀.{0,2}下移", ["glute_low"], "下臀丰满/臀峰下移(新增·试验)", ["臀后凸"], ("F", +1)),
    (r"并腿|腿.{0,2}并拢|腿.{0,2}(收拢|靠拢|夹紧)", ["legs_together"], "腿部站姿·并腿(新增·试验)", [], +1),
    (r"分开腿|腿.{0,2}(分开|张开|叉开)", ["legs_together"], "腿部站姿·并腿(新增·试验)", [], -1),
    (r"身高|个子|高度", ["height"], "身高(宏)", ["身高"], None),
    (r"胸围", ["bust_c"], "胸围", ["胸前凸", "肩/胸廓宽"], None),
    (r"胸廓|肋", ["chest_w"], "胸廓宽", ["肩/胸廓宽"], None),
    (r"胸|乳房|罩杯", ["cupsize"], "胸量(罩杯宏)", ["胸前凸", "肩/胸廓宽"], None),
    (r"肩", ["shoulder_w"], "肩宽", ["肩/胸廓宽"], None),
    (r"腰围|腰", ["waist_c"], "腰围", ["腰宽(最细)"], None),
    (r"小腹|肚子|腹", ["belly"], "小腹", ["腰宽(最细)"], None),
    (r"胯围", ["hip_c"], "胯围", ["胯宽", "臀后凸"], None),
    (r"胯厚|骨盆厚", ["hip_d"], "胯厚", ["臀后凸"], None),
    (r"翘|后凸|突出|圆润|饱满", ["glute_proj"], "臀后凸(新增·试验)", ["臀后凸"], None),
    (r"臀|屁股", ["glute_vol"], "臀体积", ["臀后凸", "胯宽"], None),
    (r"胯|髋|骨盆", ["hip_w"], "胯宽", ["胯宽"], None),
    (r"小腿.{0,3}(长|短)|小腿长", ["shin_len"], "小腿长", ["身高"], None),
    (r"大腿.{0,3}(长|短)|大腿长", ["thigh_len"], "大腿长", ["身高"], None),
    (r"腿.{0,2}(长|短)|腿长", ["thigh_len", "shin_len"], "腿长(大小腿同调)", ["身高"], None),
    (r"小腿", ["calf_c"], "小腿围", ["小腿单腿宽"], None),
    (r"大腿|腿", ["thigh_c"], "大腿围", ["大腿单腿宽"], None),
    (r"脖|颈", ["neck_len"], "颈长", [], None),
    (r"头", ["head_scale_v"], "头高(头身比)", [], None),
    (r"肌肉|壮|健硕", ["muscle"], "肌肉(宏)", ["肩/胸廓宽", "大腿单腿宽"], +1),
    (r"胖|瘦|体重", ["weight"], "体重(宏)", ["腰宽(最细)", "胯宽"], None),
]
LATER = [
    (r"乳头|胸尖|乳晕", "乳头/乳晕细节：阶段 B 目前没有对应形变（本机免费资源里没有现成的乳头目标，已如实记录为缺口；胸大形在阶段 A/B 已有）"),
    (r"指甲|脚趾甲", "阶段 D（材质 / 局部细化）"),
    (r"五官|脸|面|眼|睫|眉|鼻|嘴|唇|牙|耳", "阶段 C（头脸细化；本阶段只有头脸占位点）"),
    (r"头发|发型|刘海|发色", "阶段 D（头发）"),
    (r"皮肤|肤色|材质|纹理|指甲|唇色|瞳", "阶段 D（材质 / 局部细化）"),
]
NOUN = re.compile(r"手指|手腕|手掌|脚踝|脚腕|脚背|脚趾|上臂|大臂|前臂|小臂|肚脐|臀峰|下半球|大腿|小腿|腰围|胯宽|胯围|胯厚|胸围|肩宽|腿长|身高|个子|高度|宽度|大小|体积|长度|粗细|胸部|腰部|胯部|臀部|腿部|肩膀|"
                  r"脖子|颈部|头部|小腹|肚子|乳房|罩杯|骨盆|胸廓|屁股|胸|腰|胯|臀|腿|肩|颈|脖|头|腹|髋|肋|肌肉|体重|手|脚|足|膝")
MINUS = re.compile(r"变小|缩小|减小|更小|小|变窄|收窄|更窄|窄|变细|更细|细|收紧|收|瘦|减少|减|降低|变矮|矮|变短|缩短|更短|短|薄|变薄|扁|平|削|矮|少")
PLUS = re.compile(r"变大|加大|增大|放大|更大|大|变宽|加宽|增宽|更宽|宽|变粗|加粗|更粗|粗|增加|加|更|丰满|变高|增高|更高|高|变长|加长|更长|"
                  r"长|翘|突出|后凸|前凸|凸|抬高|提高|厚|变厚|加厚|壮|胖|圆润|饱满|拉长|拉高|多|明显")
# fraction of the half slider range per step. (Was 0.08 for 一点 = ~1 mm on the waist: a request that visibly did nothing; MPFB sliders
# have little reach, so the tiers are sized to give a perceptible but still small change.)
TIERS = [(re.compile(r"非常|特别|极|大幅|显著|十分|超"), 0.75, "很多"), (re.compile(r"明显|不少|较多|多|很|比较"), 0.50, "较多"),
         (re.compile(r"一些|适当|适度|些|较"), 0.35, "一些"), (re.compile(r"一点点|稍稍|稍微|略微|微微|微"), 0.10, "极小"),
         (re.compile(r"一点|略|稍|轻微|少许"), 0.22, "一点")]
DEFAULT_AMT = (0.12, "默认")


def _split(text):
    return [p.strip() for p in re.split(r"[，,；;。\n]|然后|并且|同时|接着|但是|但|不过|而且", text) if p.strip()]


MODE_LABEL = "规则指令模式（关键词规则解析，不是大模型；自由 AI 入口见「输入与创建」，未完成）"
PROT_RX = re.compile(r"不变|保持|别动|不要动|不要改|别改|不动|维持|不能变|不许变")
# part word -> measured metric(s) that are protected, and the control that can restore it
PROT_PARTS = [(r"身高|个子|高度", "身高", "height"), (r"胸围|胸部|胸|乳房|罩杯", "胸前凸", "cupsize"), (r"肩|胸廓", "肩/胸廓宽", "chest_w"),
              (r"腰", "腰宽(最细)", "waist_c"), (r"胯|髋|骨盆", "胯宽", "hip_c"), (r"臀|屁股", "臀后凸", "glute_proj"),
              (r"大腿", "大腿单腿宽", "thigh_c"), (r"小腿", "小腿单腿宽", "calf_c")]
PROT_TOL = {"身高": 0.005}      # metres; the rest 0.4 %H


def _todo(text, reason):
    os.makedirs(os.path.dirname(TODO_LOG), exist_ok=True)
    with open(TODO_LOG, "a", encoding="utf-8") as f:
        f.write("- %s  「%s」 → %s\n" % (time.strftime("%Y-%m-%d %H:%M"), text, reason))


def parse_nl(text):
    """-> list of dicts: ok ops {part, label, ids, sign, amount, tier, absolute, affects, clause} or refusals {clause, reason}."""
    ops = []
    for cl in _split(text):
        if PROT_RX.search(cl):
            prot = [(m, cid) for pat, m, cid in PROT_PARTS if re.search(pat, cl)]
            if prot:
                ops.append(dict(clause=cl, protect=prot))
                continue
        for pat, why in LATER:
            if re.search(pat, cl):
                ops.append(dict(clause=cl, reason="不在阶段①范围：" + why + "。已记入待办，未执行", todo=True))
                break
        else:
            rule = next((r for r in RULES if re.search(r[0], cl)), None)
            if rule is None:
                ops.append(dict(clause=cl, reason="没识别到要改的部位；现有阶段①控制里没有对应项（可能需要新增形变）", todo=True))
                continue
            pat, ids, label, aff, dflt = rule
            # "胯宽一点 / 腿长一点 / 身高多一点": the attribute itself is the direction (wider / longer / taller)
            cl_dir = re.sub(r"(胯宽|肩宽|腿长|身高)(?=\s*(?:再)?(?:一点|一些|点|些|多|明显|很|非常|稍|略|微))", r"加", cl)
            rest = NOUN.sub("", cl_dir.replace("胯宽加", "加").replace("肩宽加", "加").replace("腿长加", "加").replace("身高加", "加"))
            absm = re.search(r"(?:设为|设置为|调到|调整到|=)\s*(-?\d+(?:\.\d+)?)", cl)
            pct = re.search(r"(\d+(?:\.\d+)?)\s*%", cl)
            if isinstance(dflt, tuple):
                sign = dflt[1] * (-1 if re.search(r"减|去掉|取消|不要|少|松开|复原|回去", cl) else 1)
            else:
                sign = -1 if MINUS.search(rest) else (+1 if PLUS.search(rest) else dflt)
            if sign is None and not absm:
                ops.append(dict(clause=cl, reason="识别到「%s」，但没有方向（加大/变小/加宽/变窄…），未执行" % label))
                continue
            amount, tier, absolute = DEFAULT_AMT[0], DEFAULT_AMT[1], None
            rest = rest.replace("多一点", "一点")
            for rx, a, nm in TIERS:
                if rx.search(rest):
                    amount, tier = a, nm
                    break
            pct_height = None
            if pct:
                # N% = N% of the slider's full travel (=2N/100 of its half range); for 身高 it is N% of body height
                amount, tier = float(pct.group(1)) / 100.0 * 2.0, "%s%%(滑块行程)" % pct.group(1)
                if ids == ["height"]:
                    pct_height, tier = float(pct.group(1)) / 100.0, "%s%%(身高)" % pct.group(1)
            if absm:
                absolute, tier = float(absm.group(1)), "设为 %s" % absm.group(1)
            ops.append(dict(clause=cl, label=label, ids=ids, sign=sign, amount=amount, tier=tier, absolute=absolute,
                            affects=aff, part=re.search(pat, cl).group(0), pct_height=pct_height))
    return ops


def run_nl(text):
    obj = body()
    ops = parse_nl(text)
    if not ops:
        return ["没有内容可执行"]
    state = C.get_state(obj)
    before = measure()
    msgs, executed = [MODE_LABEL], []
    protect = {}
    for op in ops:
        for m, cid in op.get("protect", []):
            protect[m] = cid
    for op in ops:
        if "protect" in op:
            msgs.append("🔒 保护条件：「%s」→ 保持 %s 不变" % (op["clause"], "、".join(m for m, _ in op["protect"])))
            continue
        if "reason" in op:
            msgs.append("✗ 「%s」：%s" % (op["clause"], op["reason"]))
            if op.get("todo"):
                _todo(op["clause"], op["reason"])
            continue
        notes = []
        clash0 = [m for m in op["affects"][:1] if m in protect]      # primary target only; side effects are compensated below
        if clash0:
            msgs.append("✗ 「%s」与保护条件冲突：它的直接目标就是 %s（你要求保持不变），未执行。要执行请先去掉保护条件" % (op["clause"], "、".join(clash0)))
            continue
        for cid in op["ids"]:
            c = C.CTRL[cid]
            half = (c["hi"] - c["lo"]) / 2.0
            old = state[cid]
            if op["absolute"] is not None:
                new = op["absolute"]
            elif op.get("pct_height"):
                new = old + op["sign"] * op["pct_height"] * before["身高"] / HEIGHT_M_PER_UNIT
            else:
                new = old + op["sign"] * op["amount"] * half
            newc = min(max(new, c["lo"]), c["hi"])
            clamped = abs(newc - new) > 1e-9
            state[cid] = newc
            notes.append("%s %.2f→%.2f%s" % (c["label"], old, newc, "（已到该滑条的上/下限，未再放大；更大的形变超出本阶段支持范围）" if clamped else ""))
        executed.append(op)
        d = "增大" if (op["sign"] or 0) > 0 else "减小"
        msgs.append("✓ 「%s」→ 部位/属性：%s → 方向/程度：%s·%s → %s" % (
            op["clause"], op["label"], d if op["absolute"] is None else "设定", op["tier"], "；".join(notes)))
    if executed:
        C.apply_state(obj, state)
        if protect:
            msgs.extend(_restore_protected(protect, before))
            state = C.get_state(obj)
        commit(state, "自然语言：" + text)
        sync_props(bpy.context.scene, state)
        after = measure()
        aff = {m for op in executed for m in op["affects"]}
        tgt = [m for m in MET_NAMES if m in aff]
        prot = [m for m in MET_NAMES if m not in aff]
        if tgt:
            msgs.append("  目标区：" + "；".join("%s %s→%s" % (m, fmt_met(m, before[m]), fmt_met(m, after[m])) for m in tgt))
        side = []
        for m in prot:
            dv = after[m] - before[m]
            if m == "身高":
                if abs(dv) > 0.005:
                    side.append("%s %+.1f cm" % (m, dv * 100))
            elif abs(dv) > 0.004:
                side.append("%s %+.1f%%H" % (m, dv * 100))
        msgs.append("  保护区：" + ("有联动 → " + "；".join(side) if side else "腰/胸/胯/臀/腿等未列入目标的部位变化均 <0.4%H（视为未动）"))
    return msgs


def _restore_protected(protect, before):
    """After an edit, pull every protected metric back to its previous value by moving its own control (secant steps).
    Reports what was restored and what could NOT be restored (control at its limit)."""
    obj = body()
    out = []
    for m, cid in protect.items():
        tol = PROT_TOL.get(m, 0.004)
        c = C.CTRL[cid]
        st = C.get_state(obj)
        cur = measure()[m]
        drift0 = cur - before[m]
        if abs(drift0) <= tol:
            out.append("🔒 %s：变化 %s，在容差内（保持）" % (m, ("%+.1f cm" % (drift0 * 100)) if m == "身高" else ("%+.2f%%H" % (drift0 * 100))))
            continue
        x_prev, f_prev = st[cid], drift0
        step = 0.1 * (c["hi"] - c["lo"])
        x = min(max(x_prev - step * (1 if drift0 > 0 else -1) * (1 if c["kind"] != "macro" else 1), c["lo"]), c["hi"])
        ok = False
        for _ in range(8):
            st[cid] = x
            C.apply_state(obj, st)
            f = measure()[m] - before[m]
            if abs(f) <= tol:
                ok = True
                break
            if abs(f - f_prev) < 1e-9:
                break
            x_new = x - f * (x - x_prev) / (f - f_prev)
            x_prev, f_prev = x, f
            x = min(max(x_new, c["lo"]), c["hi"])
            if abs(x - x_prev) < 1e-6:
                break
        res = measure()[m] - before[m]
        fm = lambda v: ("%+.1f cm" % (v * 100)) if m == "身高" else ("%+.2f%%H" % (v * 100))
        if ok:
            out.append("🔒 %s：编辑使它变了 %s，已用「%s」拉回（现 %s，容差内）" % (m, fm(drift0), c["label"], fm(res)))
        else:
            out.append("⚠ %s：编辑使它变了 %s，调「%s」也只能拉回到 %s（控制已到极限）——保护条件未能满足，请撤销或放宽" % (m, fm(drift0), c["label"], fm(res)))
    return out


# ------------------------------------------------------------------ operators
class CS_OT_nl(bpy.types.Operator):
    bl_idname = "cs.nl_apply"
    bl_label = "执行"
    bl_description = "把自然语言解析成明确操作并真实执行（无法映射的不会假装成功）"

    def execute(self, context):
        if S.showing_before:
            self.report({'WARNING'}, "正在看「之前」，请先切回「现在」")
            return {'CANCELLED'}
        text = context.scene.cs.nl_text.strip()
        if not text:
            return {'CANCELLED'}
        log("》" + text, *run_nl(text))
        return {'FINISHED'}


class CS_OT_undo(bpy.types.Operator):
    bl_idname = "cs.undo"
    bl_label = "撤销"

    def execute(self, context):
        if S.hi <= 0:
            return {'CANCELLED'}
        S.hi -= 1
        _goto(context, S.hist[S.hi], "撤销 → " + S.hist[S.hi]["label"])
        return {'FINISHED'}


class CS_OT_redo(bpy.types.Operator):
    bl_idname = "cs.redo"
    bl_label = "重做"

    def execute(self, context):
        if S.hi >= len(S.hist) - 1:
            return {'CANCELLED'}
        S.hi += 1
        _goto(context, S.hist[S.hi], "重做 → " + S.hist[S.hi]["label"])
        return {'FINISHED'}


def _goto(context, entry, msg):
    S.showing_before = False
    obj = body()
    if entry.get("pts") is not None:
        A.restore(obj, entry["pts"])
    C.apply_state(obj, entry["state"])
    S.after = C.get_state(obj)
    S.after_pts = A.snapshot()
    sync_props(context.scene, S.after)
    log(msg)
    _health_after_commit(report=False)


class CS_OT_before(bpy.types.Operator):
    bl_idname = "cs.set_before"
    bl_label = "设定「之前」"
    mode: EnumProperty(items=[("CURRENT", "当前", ""), ("DRAFT", "交付草模", ""), ("DEFAULT", "MPFB 默认人偶", "")])

    def execute(self, context):
        obj = body()
        if S.showing_before:
            self.report({'WARNING'}, "请先切回「现在」")
            return {'CANCELLED'}
        if self.mode == "CURRENT":
            S.before, S.before_name, S.before_pts = C.get_state(obj), "此前记录的状态", A.snapshot()
        elif self.mode == "DRAFT":
            S.before, S.before_name = json.loads(obj["cs_draft_state"]), "交付草模"
            S.before_pts = dict(off=json.loads(obj.get("cs_draft_off", "{}")), mirror=True)
        else:
            S.before, S.before_name = json.loads(obj["cs_mpfb_default"]), "MPFB 默认人偶"
            S.before_pts = dict(off={}, mirror=True)
        log("「之前」= " + S.before_name)
        return {'FINISHED'}


class CS_OT_toggle(bpy.types.Operator):
    bl_idname = "cs.toggle_before"
    bl_label = "看「之前」/「现在」"

    def execute(self, context):
        obj = body()
        if S.before is None:
            S.before = json.loads(obj["cs_draft_state"])
            S.before_pts = dict(off=json.loads(obj.get("cs_draft_off", "{}")), mirror=True)
        if not S.showing_before:
            S.after = C.get_state(obj)
            S.after_pts = A.snapshot()
            S.showing_before = True
            if S.before_pts is not None:
                A.restore(obj, S.before_pts)
            C.apply_state(obj, S.before)
            sync_props(context.scene, S.before)
            log("正在显示：之前（%s）" % S.before_name)
        else:
            S.showing_before = False
            if S.after_pts is not None:
                A.restore(obj, S.after_pts)
            C.apply_state(obj, S.after)
            sync_props(context.scene, S.after)
            log("正在显示：现在")
        return {'FINISHED'}


class CS_OT_reset(bpy.types.Operator):
    bl_idname = "cs.reset"
    bl_label = "重置"
    mode: EnumProperty(items=[("DRAFT", "交付草模", ""), ("DEFAULT", "MPFB 默认", "")])

    def execute(self, context):
        obj = body()
        st = json.loads(obj["cs_draft_state" if self.mode == "DRAFT" else "cs_mpfb_default"])
        S.showing_before = False
        A.P.off = json.loads(obj.get("cs_draft_off", "{}")) if self.mode == "DRAFT" else {}
        A.save_props(obj)
        commit(st, "重置到" + ("交付草模" if self.mode == "DRAFT" else "MPFB 默认人偶"))
        sync_props(context.scene, C.get_state(obj))
        log("已重置到" + ("交付草模" if self.mode == "DRAFT" else "MPFB 默认人偶") + "（可撤销）")
        return {'FINISHED'}


class CS_OT_smooth(bpy.types.Operator):
    bl_idname = "cs.display_smooth"
    bl_label = "显示平滑"
    bl_description = "只影响显示：给人体加/去掉 Blender 自带的细分曲面修改器，让胸的多边形轮廓变圆滑。基础网格、滑块数值、测量和拟合都不变"

    def execute(self, context):
        obj = body()
        on = not C.is_display_smooth(obj)
        C.set_display_smooth(obj, on)
        log("显示平滑：" + ("开（只是显示效果，基础网格不变）" if on else "关（看到的是真实基础网格）"))
        return {'FINISHED'}


class CS_OT_view(bpy.types.Operator):
    bl_idname = "cs.view"
    bl_label = "视图"
    view: EnumProperty(items=[("FRONT", "正", ""), ("SIDE", "侧", ""), ("BACK", "背", ""), ("Q34", "3/4", "")])

    def execute(self, context):
        area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if area is None:
            return {'CANCELLED'}
        r3d = area.spaces.active.region_3d
        obj = body()
        co = C.raw_coords(obj)
        zone = C.load_zones(obj)
        keep = zone != 0
        zmid = float(0.5 * (co[keep, 2].min() + co[keep, 2].max()))
        rot = {"FRONT": (90, 0, 0), "SIDE": (90, 0, 90), "BACK": (90, 0, 180), "Q34": (85, 0, 35)}[self.view]
        from mathutils import Euler
        r3d.view_perspective = 'ORTHO'
        r3d.view_rotation = Euler([math.radians(a) for a in rot]).to_quaternion()
        r3d.view_location = (0.0, 0.0, zmid)
        r3d.view_distance = 3.2
        return {'FINISHED'}


class CS_OT_measure(bpy.types.Operator):
    bl_idname = "cs.measure"
    bl_label = "测量对照参考图"

    def execute(self, context):
        cur, ref = measure(), ref_met()
        rows = ["测量（占身高%）  当前 / 参考图 / 差"]
        for m in MET_NAMES:
            if m == "身高":
                rows.append("  身高 %.2f m（参考图绝对身高未知，仅比例有意义）" % cur[m])
            else:
                rows.append("  %-8s %5.1f / %5.1f / %+.1f" % (m, cur[m] * 100, ref[m] * 100, (cur[m] - ref[m]) * 100))
        log(*rows)
        return {'FINISHED'}


def _next_path(prefix):
    n = 0
    for f in glob.glob(os.path.join(C.VERS, prefix + "_v*.blend")):
        try:
            n = max(n, int(os.path.basename(f)[len(prefix) + 2:].split(".")[0]))
        except ValueError:
            pass
    return os.path.join(C.VERS, "%s_v%03d" % (prefix, n + 1))


class CS_OT_save(bpy.types.Operator):
    bl_idname = "cs.save_version"
    bl_label = "保存为新版本"
    bl_description = "另存为 versions/s1_draft_vNNN.blend（永不覆盖旧版本）"

    def execute(self, context):
        if S.showing_before:
            self.report({'WARNING'}, "请先切回「现在」再保存")
            return {'CANCELLED'}
        obj = body()
        base = _next_path("s1_draft")
        obj["cs_status"] = "待确认"
        obj["cs_note"] = "面板另存 %s；基于 %s；历史 %d 步" % (time.strftime("%Y-%m-%d %H:%M"), os.path.basename(bpy.data.filepath) or "未保存文件", len(S.hist))
        A.save_props(obj)
        assert not os.path.exists(base + ".blend"), "refusing to overwrite"
        bpy.ops.wm.save_as_mainfile(filepath=base + ".blend", copy=True)
        json.dump(dict(stage=STAGE_INFO[stage_of(obj)]["name"].replace("阶段 ", ""), status="待确认", saved=time.strftime("%Y-%m-%d %H:%M:%S"), state=C.get_state(obj),
                       history=[h["label"] for h in S.hist]), open(base + "_state.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        json.dump(A.export_points(obj), open(base + "_points.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        try:
            import a_ui
            json.dump(a_ui.inputs_record(obj), open(base + "_inputs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        except Exception as ex:
            log("输入记录写入失败：%s" % ex)
        log("已保存新版本：" + os.path.basename(base) + ".blend")
        return {'FINISHED'}


class CS_OT_prot_check(bpy.types.Operator):
    bl_idname = "cs.prot_check"
    bl_label = "保护检查：大比例与阶段 A 对照"
    bl_description = "把「之前」(阶段 A 已确认的草模) 与当前的大比例测量对比；超过容差的会列出（不会自动改动）"

    def execute(self, context):
        obj = body()
        cur = C.get_state(obj)
        now = measure()
        if S.before is None:
            return {'CANCELLED'}
        saved_pts = A.snapshot()
        A.restore(obj, S.before_pts)
        C.apply_state(obj, S.before)
        base = measure()
        A.restore(obj, saved_pts)
        C.apply_state(obj, cur)
        rows, bad = [], 0
        for m in MET_NAMES:
            d = now[m] - base[m]
            tol = PROT_TOL.get(m, 0.004)
            flag = abs(d) > tol
            bad += int(flag)
            rows.append("  %s %s%s" % (m, ("%+.1f cm" % (d * 100)) if m == "身高" else ("%+.2f%%H" % (d * 100)), "  ⚠超容差" if flag else ""))
        log("保护检查（现在 vs 之前=%s）：%s" % (S.before_name, "全部在容差内" if not bad else "%d 项超容差" % bad), *rows)
        return {'FINISHED'}


class CS_OT_confirm(bpy.types.Operator):
    bl_idname = "cs.confirm_stage"
    bl_label = "确认本阶段（仅用户本人点击）"
    bl_description = "只有你本人在面板点击并在弹窗里再确认才生效；程序/脚本直接调用会被拒绝。会保存一个新的已确认版本并写验收记录"
    accept_1: bpy.props.BoolProperty(name="验收项 1", default=False, options={'SKIP_SAVE'})
    accept_2: bpy.props.BoolProperty(name="验收项 2", default=False, options={'SKIP_SAVE'})
    accept_3: bpy.props.BoolProperty(name="验收项 3", default=False, options={'SKIP_SAVE'})

    def invoke(self, context, event):
        S.confirm_token = time.time()                 # set ONLY here (a UI click); not an operator property, so bpy.ops.cs.confirm_stage(...) cannot pass it
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        c = self.layout.column()
        info = STAGE_INFO[stage_of()]
        c.label(text="%s 只验收三件事（逐项勾选才能确认）：" % info["name"])
        for k, it in enumerate(info["items"], 1):
            c.prop(self, "accept_%d" % k, text=it)
        c.label(text="确认后保存新版本并写入验收记录；不会进入下一阶段")

    def execute(self, context):
        tok = getattr(S, "confirm_token", None)
        S.confirm_token = None                        # single use
        if tok is None or time.time() - tok > 600:
            log("✗ 确认被拒绝：只有用户在面板点击才能确认阶段（脚本直接调用无效）")
            self.report({'ERROR'}, "阶段确认只能由用户在面板点击完成")
            return {'CANCELLED'}
        if not (self.accept_1 and self.accept_2 and self.accept_3):
            log("✗ 未确认：三项验收内容需全部勾选")
            return {'CANCELLED'}
        obj = body()
        info = STAGE_INFO[stage_of(obj)]
        base = _next_path(info["prefix"])
        vid = os.path.basename(base)
        obj["cs_status"] = "已确认"
        obj["cs_note"] = "用户在面板确认%s %s；版本 %s；基于 %s" % (info["name"], time.strftime("%Y-%m-%d %H:%M"), vid, os.path.basename(bpy.data.filepath) or "未保存文件")
        A.save_props(obj)
        assert not os.path.exists(base + ".blend"), "refusing to overwrite"
        bpy.ops.wm.save_as_mainfile(filepath=base + ".blend", copy=True)
        rec = dict(version_id=vid, file=vid + ".blend", stage=info["name"].replace("阶段 ", ""), confirmed_by="用户（面板按钮 + 三项勾选）",
                   confirmed_at=time.strftime("%Y-%m-%d %H:%M:%S"), accepted=list(info["items"]),
                   state=C.get_state(obj), point_offsets=dict(A.P.off), mirror=A.P.mirror,
                   health=(S.health or {}).get("by_kind"), source=os.path.basename(bpy.data.filepath))
        json.dump(rec, open(base + "_state.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        rp = os.path.join(C.ROOT, "docs", "验收记录.json")
        allr = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else []
        allr.append(rec)
        json.dump(allr, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        log("%s 已确认，版本 %s（验收记录 docs/验收记录.json；不会被覆盖）" % (info["name"], vid))
        return {'FINISHED'}


def _stack(a, b, out):
    """stack two rendered sheets vertically (Blender image API only; no PIL in Blender's python)"""
    ia, ib = bpy.data.images.load(a), bpy.data.images.load(b)
    w, h = ia.size
    px = np.empty(w * h * 4 * 2, dtype=np.float32)
    ia_px = np.empty(w * h * 4, dtype=np.float32)
    ib_px = np.empty(ib.size[0] * ib.size[1] * 4, dtype=np.float32)
    ia.pixels.foreach_get(ia_px)
    ib.pixels.foreach_get(ib_px)
    px[:w * h * 4] = ib_px[:w * h * 4]             # Blender images are bottom-up: first block = lower image = "现在"
    px[w * h * 4:] = ia_px
    o = bpy.data.images.new("cs_stack", w, h * 2)
    o.pixels.foreach_set(px)
    o.filepath_raw = out
    o.file_format = 'PNG'
    o.save()
    for im in (ia, ib, o):
        bpy.data.images.remove(im)


class CS_OT_compare_render(bpy.types.Operator):
    bl_idname = "cs.compare_render"
    bl_label = "生成「之前/现在」对比图"
    bl_description = "渲染 之前/现在 的 正/侧/3-4 灰模对比图到 renders/s1_compare/（不改动当前状态）"

    def execute(self, context):
        import s1_view as V
        obj = body()
        if S.before is None:
            S.before = json.loads(obj["cs_draft_state"])
            S.before_pts = dict(off=json.loads(obj.get("cs_draft_off", "{}")), mirror=True)
        cur = S.after if S.showing_before else C.get_state(obj)
        cur_pts = S.after_pts if S.showing_before else A.snapshot()
        os.makedirs(os.path.join(C.RENDERS, "s1_compare"), exist_ok=True)
        path = os.path.join(C.RENDERS, "s1_compare", "compare_%s.png" % time.strftime("%Y%m%d_%H%M%S"))
        A.restore(obj, S.before_pts)
        before_imgs = None
        V.compare_sheet([(obj, S.before, "之前")], path.replace(".png", "_before.png"), views=("front", "side", "q34"), res=(420, 700))
        A.restore(obj, cur_pts)
        V.compare_sheet([(obj, cur, "现在")], path.replace(".png", "_after.png"), views=("front", "side", "q34"), res=(420, 700))
        _stack(path.replace(".png", "_before.png"), path.replace(".png", "_after.png"), path)
        A.restore(obj, S.before_pts if S.showing_before else cur_pts)
        C.apply_state(obj, S.before if S.showing_before else cur)
        log("对比图（上=之前，下=现在）：" + path)
        return {'FINISHED'}


# ------------------------------------------------------------------ panels
def _wrap(text, width=22):
    """The sidebar is narrow and Blender truncates long labels: split a message into several short lines."""
    out, cur, w = [], "", 0
    for ch in text:
        cw = 2 if ord(ch) > 0x2E80 else 1
        if w + cw > width * 2:
            out.append(cur)
            cur, w = "  ", 2
        cur += ch
        w += cw
    out.append(cur)
    return out


class CS_PT_main(bpy.types.Panel):
    bl_label = "身体建模（待你确认）"
    bl_idname = "CS_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = TAB
    bl_order = -100

    def draw(self, context):
        lay = self.layout
        obj = body()
        if obj is None:
            lay.label(text="没有找到人体（CS_Body）", icon='ERROR')
            return
        sc = context.scene
        box = lay.box()
        box.label(text="状态：%s　|　%s" % (obj.get("cs_status", "?"), "正在看「之前」" if S.showing_before else "正在看「现在」"))
        if stage_of(obj) == "B":
            box.label(text="阶段 B 只验收三件事：①胸部造型")
            box.label(text="②腰臀四肢造型 ③手脚造型")
            box.label(text="（五官/头发/材质/精修不验收；阶段A比例已确认）")
            box.label(text="大比例对照：" + ("见下方「保护检查」" if S.log else "未检查"))
            box.operator("cs.prot_check", icon='VIEWZOOM')
        else:
            box.label(text="只验收三件事：①整体高矮胖瘦")
            box.label(text="②肩腰胯与腿长比例 ③胸臀大体大小/位置")
            box.label(text="（手足/五官/头发/材质不验收）")
        row = lay.row(align=True)
        for v, t in (("FRONT", "正"), ("SIDE", "侧"), ("BACK", "背"), ("Q34", "3/4")):
            row.operator("cs.view", text=t).view = v
        lay.prop(sc.cs, "shade", expand=True)
        lay.operator("cs.display_smooth", icon='MOD_SUBSURF',
                     text="显示平滑：%s（仅显示）" % ("开" if C.is_display_smooth(obj) else "关"))
        box = lay.box()
        row = box.row(align=True)
        row.operator("cs.undo", icon='LOOP_BACK')
        row.operator("cs.redo", icon='LOOP_FORWARDS')
        box.label(text="历史 %d/%d：%s" % (S.hi + 1, len(S.hist), S.hist[S.hi]["label"] if 0 <= S.hi < len(S.hist) else "-"))
        box.operator("cs.toggle_before", icon='ARROW_LEFTRIGHT', text="看「现在」" if S.showing_before else "看「之前」（%s）" % S.before_name)
        row = box.row(align=True)
        row.operator("cs.set_before", text="记当前为之前").mode = "CURRENT"
        row.operator("cs.set_before", text="之前=草模").mode = "DRAFT"
        row.operator("cs.set_before", text="之前=默认人偶").mode = "DEFAULT"
        box.operator("cs.compare_render", icon='RENDER_STILL')
        box = lay.box()
        box.label(text="自然语言修改（规则指令模式，真实执行）：")
        box.prop(sc.cs, "nl_text")
        box.operator("cs.nl_apply", icon='PLAY')
        for ln in S.log[-12:]:
            for part in _wrap(ln):
                box.label(text=part)
        box = lay.box()
        row = box.row(align=True)
        row.operator("cs.reset", text="重置到草模").mode = "DRAFT"
        row.operator("cs.reset", text="重置到默认人偶").mode = "DEFAULT"
        box.operator("cs.measure", icon='DRIVER_DISTANCE')
        box.operator("cs.save_version", icon='FILE_TICK')
        box.operator("cs.confirm_stage", icon='CHECKMARK')
        box.label(text="点工具内 Ctrl+Z 撤销；也可用面板按钮" if obj.get('v5_interaction') else "撤销请用本面板按钮，别用 Ctrl+Z")


GROUPS = ["整体", "上身", "胸", "腰腹", "胯臀", "腿", "头颈", "试验·新增",
          "阶段B·胸", "阶段B·腰腹", "阶段B·臀", "阶段B·四肢", "阶段B·手", "阶段B·脚", "阶段B·细化", "阶段B·姿势",
          "红·胸造型", "红·臀造型"]
RED_PANEL = {"红·胸造型": ("bust", ("水滴", "圆形", "上翘")), "红·臀造型": ("glute", ("蜜桃", "心形", "方形"))}


class CS_OT_red_preset(bpy.types.Operator):
    bl_idname = "cs.red_preset"
    bl_label = "红点造型预设"
    bl_description = "一次设好这个部位的全部红滑块（只动红滑块，绿点比例/白点体态不变；可撤销）；“原样”= 全部归 0"
    part: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        obj = body()
        if obj is None:
            return {'CANCELLED'}
        S.showing_before = False
        st = dict(C.get_state(obj))
        st.update(U.preset_state(self.part, self.name))
        full = commit(st, "红点造型：%s·%s" % ("胸" if self.part == "bust" else "臀", self.name))
        sync_props(context.scene, full)
        log("红点造型：%s 已设为「%s」（只改了红滑块，可撤销）" % ("胸" if self.part == "bust" else "臀", self.name))
        return {'FINISHED'}


def _make_group_panel(g):
    def draw(self, context):
        if g in RED_PANEL:
            part, names = RED_PANEL[g]
            row = self.layout.row(align=True)
            row.enabled = not S.showing_before
            for nm in names + ("原样",):
                op = row.operator("cs.red_preset", text=nm)
                op.part, op.name = part, nm
        col = self.layout.column(align=True)
        col.enabled = not S.showing_before
        for c in exposed():
            if c["group"] == g:
                tag = "" if c["have"] == "已有" else "【新增】"
                col.prop(context.scene.cs, "c_" + c["id"], text=tag + c["label"], slider=True)
        if g == "胸":
            self.layout.label(text="乳头/尖度细节属于阶段②；胸的圆润/下缘见「试验·新增」")
        if g == "试验·新增":
            self.layout.label(text="试验形变：在副本上验证，你确认后才入库")
        if g == "阶段B·细化":
            self.layout.label(text="只修尖点/折痕，最大 1 cm，不改大比例")
        if g in RED_PANEL:
            self.layout.label(text="红点=胸/臀造型：只改形状，不改绿点比例和白点体态")
            self.layout.label(text="上面一排是一键预设，之后还能逐个微调")
        if g == "阶段B·姿势":
            self.layout.label(text="只是姿势：拉回 0 = 原来的 A 字站姿，体型不变")
        elif g.startswith("阶段B"):
            self.layout.label(text="范围已按本机扫描限幅")
    return type("CS_PT_g_%d" % GROUPS.index(g), (bpy.types.Panel,), dict(
        bl_label=g, bl_space_type='VIEW_3D', bl_region_type='UI', bl_category=TAB, bl_parent_id="CS_PT_main",
        bl_options={'DEFAULT_CLOSED'} if g in ("头颈", "腰腹", "试验·新增", "整体", "上身", "胯臀", "腿") else set(), draw=draw))


CLASSES = []


def boot():
    """Called from s1_panel_boot.py (GUI) or from tests (background)."""
    C.enable_mpfb()
    U.register()
    props = make_props()
    CS_PT_main.bl_label = STAGE_INFO[stage_of()]["title"]
    CLASSES[:] = [props, CS_OT_nl, CS_OT_undo, CS_OT_redo, CS_OT_before, CS_OT_toggle, CS_OT_reset, CS_OT_view, CS_OT_smooth,
                  CS_OT_measure, CS_OT_save, CS_OT_confirm, CS_OT_prot_check, CS_OT_compare_render, CS_OT_red_preset, CS_PT_main] + [_make_group_panel(g) for g in GROUPS]
    for cl in CLASSES:
        bpy.utils.register_class(cl)
    bpy.types.Scene.cs = PointerProperty(type=props)
    obj = body()
    if obj is None:
        return
    if "cs_points" not in obj:
        A.build_registry(obj)
    A.load(obj)
    import a_ui
    a_ui.register()
    if stage_of(obj) == "B":
        A.P.show_green = False                   # stage B shows the white points by default (green points: toggle in the points panel)
    st = C.get_state(obj)
    C.apply_state(obj, st)                       # re-apply so that the stored point offsets are part of the mesh
    S.hist = [dict(label="打开文件时的状态", state=dict(st), t=time.time(), co=None, pts=A.snapshot())]
    S.hi = 0
    _health_after_commit()
    S.after = dict(st)
    S.after_pts = A.snapshot()
    S.before = json.loads(obj["cs_draft_state"]) if "cs_draft_state" in obj else dict(st)
    S.before_pts = dict(off=json.loads(obj.get("cs_draft_off", "{}")), mirror=True)
    S.before_name = "阶段A已确认草模(v016)" if stage_of(obj) == "B" else "交付草模"
    for sc in bpy.data.scenes:
        sync_props(sc, st)
    log("已就绪：拖滑块或在下方输入一句话，例如「胯部加宽一点」")

    def _init_view():
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.spaces.active.show_region_ui = True
                    with bpy.context.temp_override(window=win, area=area, region=next(r for r in area.regions if r.type == 'WINDOW')):
                        bpy.ops.cs.view(view="FRONT")
                    apply_shading(win.screen, "CLAY")
        return None
    if not bpy.app.background:
        bpy.app.timers.register(_init_view, first_interval=0.8)


def unregister():
    try:
        import a_ui
        a_ui.unregister()
    except Exception:
        pass
    for cl in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cl)
        except RuntimeError:
            pass
    if hasattr(bpy.types.Scene, "cs"):
        del bpy.types.Scene.cs
