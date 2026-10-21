"""v3 panel (A3 结构人体与模块化零件): runtime overlay on s1_panel / a_ui (those files are unchanged).

register()   - BEFORE s1_panel.boot(): stage "A3" (its controls get panel properties), stage info / main-panel text, prompt and
               preset wrappers (v3 structure + part design are kept), and the panel classes are prepared.
after_boot() - AFTER s1_panel.boot(): registers the v3 sub-panels, sets 「之前」= v020.

Honest notes shown in the panel:
  * the prompt entry is a word-table RULE system (not an AI model);
  * install sliders and install-point drags are health-guarded: a change that would add a spike / crease is scaled back and
    the message says so (never silently);
  * 胖瘦 outside -0.5..1 is refused with a message (never clamped).
"""
import os
import json
import time

import bpy
from bpy.props import FloatProperty, BoolProperty, EnumProperty, StringProperty, PointerProperty

import s1_core as C
import a_points as A
import a_inputs as I
import s1_panel as PN
import v3_body as VB
import v3_parts as PT
import v3_inputs as VI

S = PN.S
TAB = PN.TAB
STRUCT_KEYS = ("v3_bustc_back", "v3_back_narrow", "back_anatomy", "back_relax", "back_slim", "v3_relief",
               "v3_parts_on", "v3_bust_on", "v3_glute_on")
INSTALL_KEYS = tuple(PT.cid(p, ax) for p in PT.PARTS for ax, *_ in PT.AX)
STAGE_A3 = dict(title="v3 结构人体与模块化零件（待你确认）", name="v3 结构人体与模块化零件", prefix="s1_confirmedA3",
                items=["① 整体体型 + 胸臀零件组合效果", "② 身体结构与零件连接自然", "③ 绿点/零件操作符合预期，撤销/保存可靠"])
UI3 = dict(fat_err="", girths=None, last_msgs=[])
_ORIG = {}


# ------------------------------------------------------------------ stage
def stage_of(obj=None):
    obj = obj if obj is not None else PN.body()
    if obj is not None and str(obj.get("cs_stage", "")).startswith("A3"):
        return "A3"
    return _ORIG["stage_of"](obj)


def _keep_v3(st, cur, saved_keys=()):
    """v3 structure + part install values come from the current body unless `saved_keys` has them"""
    for k in STRUCT_KEYS + INSTALL_KEYS:
        if k in cur and k not in saved_keys:
            st[k] = cur[k]
    return st


# ------------------------------------------------------------------ prompt / presets (wrappers)
def state_from_understanding(und, base=None):
    if base is None:
        obj = PN.body()
        base = C.default_state()
        if obj is not None:
            _keep_v3(base, C.get_state(obj))
    return _ORIG["state_from_understanding"](und, base)


def apply_prompt(text):
    und, full = _ORIG["apply_prompt"](text)
    if full is not None:
        PN.log("ℹ v3：身体结构设置（后背/起伏）、胸臀零件安装和零件编辑保留不变；只改了上面认出的项")
    return und, full


def apply_preset(key):
    import a_ui
    obj = PN.body()
    name, st, off, mir = a_ui._preset_data(key)
    parts, saved = None, ()
    if key.startswith("U"):
        data = I.user_presets()[int(key[1:])][2]
        parts, saved = data.get("parts"), set(data.get("state", {}))
    st = _keep_v3(dict(st), C.get_state(obj), saved)
    I.apply_preset_data(obj, st, off, mir, parts=parts)
    full = PN.commit(C.get_state(obj), "预设：" + name)
    PN.sync_props(bpy.context.scene, full)
    PN.log("已应用预设「%s」（可撤销）%s" % (name, "；含零件安装/编辑" if parts else "；v3 结构设置和零件保持当前"))
    return full


# ------------------------------------------------------------------ properties
def _fat_get(self):
    obj = PN.body()
    return float(VB.state_to_fat(C.get_state(obj))) if obj is not None else 0.0


def _fat_set(self, v):
    if S.sync or S.showing_before or PN.body() is None:
        return
    obj = PN.body()
    try:
        st = VI.set_fat(C.get_state(obj), v)
    except ValueError as ex:
        UI3["fat_err"] = "✗ %s：没有执行（不会悄悄截断）" % ex
        PN.log(UI3["fat_err"])
        return
    UI3["fat_err"] = ""
    full = PN.commit(st, "胖瘦 → %.2f" % v, coalesce="v3_fat")
    PN.sync_props(bpy.context.scene, full)


def _side(self):
    return self.side.lower()


def _inst_get(kind, ax):
    def g(self):
        obj = PN.body()
        return float(C.get_state(obj).get(PT.cid(kind + "_" + _side(self), ax), 0.0)) if obj is not None else 0.0
    return g


def install_change(obj, part, values, mirror=True, label=None):
    """set install values {axis: value} of `part` (+ mirror) with the health guard, commit.  Returns (state, messages)."""
    st0 = dict(C.get_state(obj))
    st = dict(st0)
    for ax, v in values.items():
        lo, hi = next((a[3], a[4]) for a in PT.AX if a[0] == ax)
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            return st0, ["✗ 安装量 %s=%.2f 超出 %.1f～%.1f：没有执行" % (ax, v, lo, hi)]
        st[PT.cid(part, ax)] = float(v)
    if mirror:
        PT._mirror_install(st, part)
    h0 = S.health if (S.health_on and S.health is not None) else None
    st2, msgs = PT.guarded_apply(obj, st, st0, h0=h0)
    PN.commit(st2, label or ("零件安装：" + part), coalesce="v3_inst_" + part + "_" + "_".join(values))
    return st2, msgs


def _inst_set(kind, ax):
    def s(self, v):
        if S.sync or S.showing_before or PN.body() is None:
            return
        obj = PN.body()
        p = kind + "_" + _side(self)
        lab = next(a[1] for a in PT.AX if a[0] == ax)
        full, msgs = install_change(obj, p, {ax: v}, mirror=self.mirror, label="零件安装：%s %s" % (PT.part_obj_name(p), lab))
        PN.sync_props(bpy.context.scene, full)
        if msgs:
            PN.log(*msgs)
    return s


_SRC_ITEMS = {}     # Blender requires Python to keep the dynamic enum strings alive (otherwise non-ASCII names come back as garbage)


def _src_items(kind):
    def f(self, context):
        names = PT.list_sources(kind)
        items = [(n, n, "同拓扑零件来源 assets/parts/%s_%s.json" % (kind, n)) for n in names] or [("", "（还没有保存的来源）", "")]
        if _SRC_ITEMS.get(kind) != items:
            _SRC_ITEMS[kind] = items
        return _SRC_ITEMS[kind]
    return f


def _props():
    ann = dict(
        fat=FloatProperty(name="胖瘦", get=_fat_get, set=_fat_set, soft_min=VB.FAT_LO, soft_max=VB.FAT_HI, step=5, precision=2,
                          description="一个数：-0.5（极瘦，真实形变）～0（偏瘦）～0.4（匀称）～1（肥胖）。可直接输入数字；超出范围会提示并拒绝，不会悄悄截断"),
        mirror=BoolProperty(name="左右镜像", default=True, description="改一侧零件安装时另一侧同步（默认开）"),
        side=EnumProperty(name="侧", items=[("L", "左", "角色左侧"), ("R", "右", "角色右侧")], default="L"),
        src_name=StringProperty(name="来源名", default="我的造型"),
        src_bust=EnumProperty(name="胸来源", items=_src_items("bust")),
        src_glute=EnumProperty(name="臀来源", items=_src_items("glute")),
    )
    for k in PT.KINDS:
        for ax, lab, unit, lo, hi in PT.AX:
            ann["i_%s_%s" % (k, ax)] = FloatProperty(name=lab + (("（%s）" % unit) if unit else ""), min=lo, max=hi, precision=2,
                                                      step=10 if unit == "cm" else (50 if unit else 1),
                                                      get=_inst_get(k, ax), set=_inst_set(k, ax),
                                                      description="零件自身坐标系；接口环不动。带健康守护：会新增尖刺/折痕时自动收回并提示")
    return type("CSV3Props", (bpy.types.PropertyGroup,), {"__annotations__": ann})


# ------------------------------------------------------------------ operators
class CS_OT_v3_type(bpy.types.Operator):
    bl_idname = "cs.v3_body_type"
    bl_label = "体型"
    bl_description = "只改胖瘦 / 肌肉 / 腹部 / 大腿·上臂·小腿脂肪；比例、胸臀造型、绿点、零件都保留；可撤销"
    name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        obj = PN.body()
        if obj is None or S.showing_before:
            return {'CANCELLED'}
        st = VI.body_type_state(C.get_state(obj), self.name)
        full = PN.commit(st, "体型：" + self.name)
        PN.sync_props(context.scene, full)
        PN.log("体型「%s」：%s（只改胖瘦/肌肉/四肢腹部脂肪，可撤销）" % (self.name, VB.fat_effective_text(full)))
        return {'FINISHED'}


class CS_OT_v3_girth(bpy.types.Operator):
    bl_idname = "cs.v3_girths"
    bl_label = "测量围度（真实截面）"
    bl_description = "用水平面真实切开网格，量卷尺式周长（跨过乳沟/脊柱沟，和皮尺一样）"

    def execute(self, context):
        import v3_measure as VM
        obj = PN.body()
        g = VM.Girths(obj, C.load_zones(obj))(C.raw_coords(obj))
        UI3["girths"] = g
        PN.log("围度（真实闭合截面·卷尺式，mm）：身高 %.0f  胸围 %.0f  下胸围 %.0f  腰围 %.0f  臀围 %.0f  大腿围 %.0f" % (
            g["H"], g["bust"], g["underbust"], g["waist"], g["hip"], g["thigh"]))
        return {'FINISHED'}


def _parts_of(kind, side, mirror):
    p = kind + "_" + side
    return [p, PT._mirror_part(p)] if mirror else [p]


class CS_OT_v3_show(bpy.types.Operator):
    bl_idname = "cs.v3_part_show"
    bl_label = "显示/隐藏零件"
    bl_description = "只影响显示：隐藏零件时身体在该处的面重新显示（形状相同，不是洞），零件数据保留"
    kind: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    side: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        obj = PN.body()
        po = PT.part_objects()[self.kind + "_" + self.side]
        if po is None:
            return {'CANCELLED'}
        on = po.hide_get()
        for p in [self.kind + "_" + self.side]:
            PT.set_visible(obj, p, on)
        PN.log("%s：%s" % (PT.part_obj_name(self.kind + "_" + self.side), "显示" if on else "隐藏（身体在该处的面显示出来）"))
        return {'FINISHED'}


class CS_OT_v3_select(bpy.types.Operator):
    bl_idname = "cs.v3_part_select"
    bl_label = "选中零件"
    bl_description = "选中这个零件物体，之后可按 Tab 进入编辑模式修改顶点；改完 Tab 退出，再点「采纳零件编辑」"
    kind: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    side: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        p = self.kind + "_" + self.side
        po = PT.part_objects()[p]
        if po is None:
            return {'CANCELLED'}
        if context.object is not None and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if po.hide_get():
            PT.set_visible(PN.body(), p, True)
        for o in context.view_layer.objects:
            o.select_set(False)
        po.select_set(True)
        context.view_layer.objects.active = po
        PN.log("已选中 %s：Tab 进编辑模式改顶点（接口环顶点属于身体，改了会被复原），改完 Tab 退出，点「采纳零件编辑」" % po.name)
        return {'FINISHED'}


def _flush_edit_mode(context):
    if context.object is not None and context.object.mode == 'EDIT' and context.object.get("v3_part_id"):
        bpy.ops.object.mode_set(mode='OBJECT')


class CS_OT_v3_capture(bpy.types.Operator):
    bl_idname = "cs.v3_part_capture"
    bl_label = "采纳零件编辑"
    bl_description = "把零件物体上的手工改动记入零件形状（存在零件自身坐标系里，随体型变化跟随）；进入撤销历史"

    def execute(self, context):
        _flush_edit_mode(context)
        obj = PN.body()
        pend = PT.pending_edits(obj)
        if not pend:
            PN.log("没有发现未采纳的零件编辑")
            return {'CANCELLED'}
        PT._ORIG["log"] = []
        full = PN.commit(C.get_state(obj), "零件手工编辑：" + "、".join(PT.part_obj_name(p) for p in pend))
        PN.log(*(PT._ORIG.pop("log", []) or ["零件编辑已采纳"]))
        PN.sync_props(context.scene, full)
        return {'FINISHED'}


class CS_OT_v3_edit_mirror(bpy.types.Operator):
    bl_idname = "cs.v3_part_edit_mirror"
    bl_label = "编辑镜像到另一侧"
    bl_description = "把这一侧零件的手工编辑 / 替换来源复制到另一侧（镜像），可撤销"
    kind: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    side: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        _flush_edit_mode(context)
        obj = PN.body()
        p = self.kind + "_" + self.side
        q = PT._mirror_part(p)
        mv = A.P.mvmap
        PT.PD[q] = dict(e={str(int(mv[int(v)])): list(x) for v, x in PT.PD[p]["e"].items()}, source=PT.PD[p]["source"] + "（镜像）")
        PT.save_data(obj)
        full = PN.commit(C.get_state(obj), "零件编辑镜像：%s → %s" % (PT.part_obj_name(p), PT.part_obj_name(q)))
        PN.sync_props(context.scene, full)
        PN.log("%s 的形状已镜像到 %s（可撤销）" % (PT.part_obj_name(p), PT.part_obj_name(q)))
        return {'FINISHED'}


class CS_OT_v3_clear(bpy.types.Operator):
    bl_idname = "cs.v3_part_clear"
    bl_label = "零件形状复原"
    bl_description = "清除这个零件的手工编辑 / 替换来源（回到身体自身形状；安装量不变）；镜像开时两侧一起；可撤销"
    kind: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    side: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        _flush_edit_mode(context)
        obj = PN.body()
        ps = _parts_of(self.kind, self.side, context.scene.cs_v3.mirror)
        PT.write_parts(obj)                      # drop unsaved hand edits on these part objects
        for p in ps:
            PT.clear_edits(obj, p)
        full = PN.commit(C.get_state(obj), "零件形状复原：" + "、".join(PT.part_obj_name(p) for p in ps))
        PN.sync_props(context.scene, full)
        PN.log("零件形状已复原（可撤销）：" + "、".join(PT.part_obj_name(p) for p in ps))
        return {'FINISHED'}


class CS_OT_v3_src_apply(bpy.types.Operator):
    bl_idname = "cs.v3_part_source"
    bl_label = "替换为所选来源"
    bl_description = "用同拓扑的零件来源替换零件形状（镜像开时两侧一起）；可撤销。任意外来网格导入未实现"
    kind: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        _flush_edit_mode(context)
        obj = PN.body()
        v = context.scene.cs_v3
        name = getattr(v, "src_" + self.kind)
        if not name:
            PN.log("还没有可用的%s零件来源（先用「存为来源」保存一个）" % PT.KINDS[self.kind]["label"])
            return {'CANCELLED'}
        try:
            PT.write_parts(obj)
            done = PT.apply_source(obj, self.kind + "_" + v.side.lower(), name, mirror=v.mirror)
        except (ValueError, OSError) as ex:
            PN.log("✗ 替换失败：%s" % ex)
            return {'CANCELLED'}
        full = PN.commit(C.get_state(obj), "零件替换：%s → %s" % ("、".join(PT.part_obj_name(p) for p in done), name))
        PN.sync_props(context.scene, full)
        PN.log("已替换 %s 为来源「%s」（可撤销）" % ("、".join(PT.part_obj_name(p) for p in done), name))
        return {'FINISHED'}


class CS_OT_v3_src_save(bpy.types.Operator):
    bl_idname = "cs.v3_part_source_save"
    bl_label = "存为来源"
    bl_description = "把这一侧零件当前形状存成可替换来源 assets/parts/<部位>_<名称>.json（重名自动加序号，不覆盖）"
    kind: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        _flush_edit_mode(context)
        obj = PN.body()
        v = context.scene.cs_v3
        if PT.pending_edits(obj):
            PN.log("先点「采纳零件编辑」，再存为来源")
            return {'CANCELLED'}
        base = "".join(ch for ch in v.src_name.strip() if ch not in '\\/:*?"<>|') or "造型"
        name, n = base, 1
        while os.path.exists(PT.source_path(self.kind, name)):
            n += 1
            name = "%s_%d" % (base, n)
        path = PT.save_source(obj, self.kind + "_" + v.side.lower(), name, note="面板保存 " + time.strftime("%Y-%m-%d %H:%M"))
        PN.log("已保存零件来源：" + os.path.relpath(path, C.ROOT))
        return {'FINISHED'}


class CS_OT_v3_composite(bpy.types.Operator):
    bl_idname = "cs.v3_composite"
    bl_label = "生成焊接合成副本"
    bl_description = "生成一个真正焊接的单一网格（身体 + 当前零件形状，零件面粉/橙色）；原身体和零件不变；是快照，改动后需重建"

    def execute(self, context):
        _flush_edit_mode(context)
        obj = PN.body()
        rep = PT.build_composite(obj)
        ie = rep["interface_edges"]
        PN.log("合成副本 %s：%d 顶点 / %d 面；焊接=%s（接口边 胸 %d / 臀 %d 条都被零件面和身体面共用，零件开口边 %d，"
               "被3个以上面共用的边 %d）；与身体顶点最大偏差 %.4f mm（快照，改动后需重建）" % (
            rep["object"], rep["n_verts"], rep["n_faces"], "是" if rep["welded"] else "否", ie["bust"]["part_body_shared"],
            ie["glute"]["part_body_shared"], ie["bust"]["part_open"] + ie["glute"]["part_open"], rep["edges_used_by_more_than_2_faces"],
            rep["max_vertex_offset_vs_body_mm"]))
        return {'FINISHED'}


class CS_OT_v3_composite_show(bpy.types.Operator):
    bl_idname = "cs.v3_composite_show"
    bl_label = "看合成副本 / 看分件"
    bl_description = "只切换显示：合成副本（一个焊接网格） ↔ 身体 + 零件分件"

    def execute(self, context):
        _flush_edit_mode(context)
        obj = PN.body()
        ob = bpy.data.objects.get(PT.COMP)
        if ob is None:
            PN.log("还没有合成副本：先点「生成焊接合成副本」")
            return {'CANCELLED'}
        on = ob.hide_get()
        PT.show_composite(obj, on)
        PN.log("显示：" + ("合成副本（单一焊接网格；快照）" if on else "身体 + 零件分件"))
        return {'FINISHED'}


# ------------------------------------------------------------------ panels
def draw_main(self, context):
    """CS_PT_main.draw with the A3 checklist (stage A / B files keep the original panel)"""
    if PN.stage_of() != "A3":
        return _ORIG["main_draw"](self, context)
    lay = self.layout
    obj = PN.body()
    if obj is None:
        lay.label(text="没有找到人体（CS_Body）", icon='ERROR')
        return
    sc = context.scene
    box = lay.box()
    box.label(text="状态：%s　|　%s" % (obj.get("cs_status", "?"), "正在看「之前」" if S.showing_before else "正在看「现在」"))
    box.label(text="v3 只验收三件事：")
    for it in STAGE_A3["items"]:
        box.label(text=it)
    box.label(text="（阶段A比例已确认；五官/头发/材质不验收）")
    box.operator("cs.prot_check", icon='VIEWZOOM', text="保护检查：大比例与「之前」(v020) 对照")
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
    box.operator("cs.compare_render", icon='RENDER_STILL')
    box = lay.box()
    box.label(text="自然语言修改（规则指令模式，不是大模型）：")
    box.prop(sc.cs, "nl_text")
    box.operator("cs.nl_apply", icon='PLAY')
    for ln in S.log[-12:]:
        for part in PN._wrap(ln):
            box.label(text=part)
    box = lay.box()
    row = box.row(align=True)
    row.operator("cs.reset", text="重置到交付草模").mode = "DRAFT"
    row.operator("cs.reset", text="重置到默认人偶").mode = "DEFAULT"
    box.operator("cs.measure", icon='DRIVER_DISTANCE')
    box.operator("cs.save_version", icon='FILE_TICK')
    box.operator("cs.confirm_stage", icon='CHECKMARK')
    box.label(text="点工具内 Ctrl+Z 撤销；也可用面板按钮" if obj.get('v5_interaction') else "撤销请用本面板按钮，别用 Ctrl+Z")


class _Sub:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = TAB
    bl_parent_id = "CS_PT_main"


class CS_PT_v3_body(_Sub, bpy.types.Panel):
    bl_label = "v3 胖瘦 / 体型 / 身体结构"
    bl_order = -3

    def draw(self, context):
        lay = self.layout
        obj = PN.body()
        if obj is None:
            return
        v = context.scene.cs_v3
        st = C.get_state(obj)
        col = lay.column(align=True)
        col.enabled = not S.showing_before
        col.prop(v, "fat", slider=False)
        col.label(text="-0.5 极瘦 · 0 偏瘦 · 0.4 匀称 · 1 肥胖（可直接输入）")
        for part in PN._wrap("实际生效：" + VB.fat_effective_text(st), 30):
            col.label(text=part)
        if UI3["fat_err"]:
            col.label(text=UI3["fat_err"], icon='ERROR')
        lay.label(text="体型（只改胖瘦/肌肉/腹/四肢脂肪）：")
        names = [n for n, _, _ in VI.BODY_TYPES]
        for chunk in (names[:3], names[3:]):
            row = lay.row(align=True)
            row.enabled = not S.showing_before
            for n in chunk:
                row.operator("cs.v3_body_type", text=n).name = n
        lay.operator("cs.v3_girths", icon='DRIVER_DISTANCE')
        g = UI3["girths"]
        if g:
            lay.label(text="胸 %.0f  下胸 %.0f  腰 %.0f  臀 %.0f  大腿 %.0f mm" % (g["bust"], g["underbust"], g["waist"], g["hip"], g["thigh"]))
        lay.label(text="肌肉在「整体」组，肩宽在「上身」组，两者分开")
        box = lay.box()
        box.label(text="身体结构（v3）：")
        c2 = box.column(align=True)
        c2.enabled = not S.showing_before
        for k in ("v3_relief", "v3_bustc_back", "v3_back_narrow"):
            if hasattr(context.scene.cs, "c_" + k):
                c2.prop(context.scene.cs, "c_" + k, text=C.CTRL[k]["label"], slider=True)
        box.label(text="脂肪分布强度（随胖瘦自动）：%.2f" % float(st.get("v3_fatdist", 0.0)))


class CS_PT_v3_parts(_Sub, bpy.types.Panel):
    bl_label = "v3 胸 / 臀零件（粉=胸，橙=臀）"
    bl_order = -2

    def draw(self, context):
        lay = self.layout
        obj = PN.body()
        if obj is None or PT.PROP not in obj:
            lay.label(text="这个文件没有零件系统")
            return
        sc = context.scene
        v = sc.cs_v3
        top = lay.column(align=True)
        top.enabled = not S.showing_before
        if hasattr(sc.cs, "c_v3_parts_on"):
            top.prop(sc.cs, "c_v3_parts_on", text="零件系统生效（0 = 与 v020 相同）", slider=True)
        row = lay.row(align=True)
        row.prop(v, "mirror")
        row.prop(v, "side", expand=True)
        lay.label(text="安装点 = 绿点面板里的「零件安装点」（拖动=移位）")
        pend = {}
        for p, po in PT.part_objects().items():
            if po is not None and po.mode != 'EDIT':
                try:
                    cur, fit, _ = PT._read_part(po)
                    m = float(abs(cur - fit).max()) if len(cur) else 0.0
                    if m > PT.CAPTURE_TOL:
                        pend[p] = m
                except Exception:
                    pass
            elif po is not None:
                pend[p] = -1.0
        sd = v.side.lower()
        for k, kd in PT.KINDS.items():
            box = lay.box()
            p = k + "_" + sd
            po = PT.part_objects()[p]
            hdr = box.row(align=True)
            hdr.label(text="%s零件（%s）" % (kd["label"], "粉" if k == "bust" else "橙"))
            if hasattr(sc.cs, "c_v3_%s_on" % k):
                hdr.prop(sc.cs, "c_v3_%s_on" % k, text="形状生效", slider=True)
            row = box.row(align=True)
            for s in ("l", "r"):
                q = PT.part_objects()[k + "_" + s]
                if q is None:
                    continue
                op = row.operator("cs.v3_part_show", text=("左" if s == "l" else "右") + ("：显示中" if not q.hide_get() else "：已隐藏"),
                                  icon='HIDE_OFF' if not q.hide_get() else 'HIDE_ON')
                op.kind, op.side = k, s
                op = row.operator("cs.v3_part_select", text="选中", icon='RESTRICT_SELECT_OFF')
                op.kind, op.side = k, s
            col = box.column(align=True)
            col.enabled = not S.showing_before
            col.label(text="安装（%s侧%s）：" % ("左" if sd == "l" else "右", "，镜像同步" if v.mirror else ""))
            for ax, *_ in PT.AX:
                col.prop(v, "i_%s_%s" % (k, ax), slider=True)
            box.label(text="来源：" + PT.PD.get(p, {}).get("source", "?"))
            for q in (k + "_l", k + "_r"):
                if q in pend:
                    box.label(text="%s 有未采纳编辑%s" % (PT.part_obj_name(q), "" if pend[q] < 0 else "（%.1f mm）" % (pend[q] * 1000)), icon='INFO')
            row = box.row(align=True)
            row.operator("cs.v3_part_capture", text="采纳零件编辑")
            op = row.operator("cs.v3_part_edit_mirror", text="镜像到另一侧")
            op.kind, op.side = k, sd
            op = row.operator("cs.v3_part_clear", text="复原形状")
            op.kind, op.side = k, sd
            row = box.row(align=True)
            row.prop(v, "src_" + k, text="")
            row.operator("cs.v3_part_source", text="替换").kind = k
            row = box.row(align=True)
            row.prop(v, "src_name", text="")
            row.operator("cs.v3_part_source_save", text="存为来源").kind = k
        box = lay.box()
        box.label(text="合成副本（真正焊接的单一网格）：")
        row = box.row(align=True)
        row.operator("cs.v3_composite", icon='MESH_DATA')
        row.operator("cs.v3_composite_show", icon='ARROW_LEFTRIGHT')
        lay.label(text="零件安装和拖动带健康守护：会新增尖刺/折痕时自动收回并提示")


OPS = [CS_OT_v3_type, CS_OT_v3_girth, CS_OT_v3_show, CS_OT_v3_select, CS_OT_v3_capture, CS_OT_v3_edit_mirror, CS_OT_v3_clear,
       CS_OT_v3_src_apply, CS_OT_v3_src_save, CS_OT_v3_composite, CS_OT_v3_composite_show]
PANELS = [CS_PT_v3_body, CS_PT_v3_parts]
_REG = []


# ------------------------------------------------------------------ registration
def register():
    """call BEFORE s1_panel.boot()"""
    if _ORIG:
        return
    import a_ui
    if "A3" not in PN.STAGE_CTRL:
        PN.STAGE_CTRL = tuple(PN.STAGE_CTRL) + ("A3",)
    PN.STAGE_INFO["A3"] = STAGE_A3
    _ORIG.update(stage_of=PN.stage_of, main_draw=PN.CS_PT_main.draw, state_from_understanding=I.state_from_understanding,
                 apply_prompt=a_ui.apply_prompt, apply_preset=a_ui.apply_preset)
    PN.stage_of = stage_of
    PN.CS_PT_main.draw = draw_main
    I.state_from_understanding = state_from_understanding
    a_ui.apply_prompt = apply_prompt
    a_ui.apply_preset = apply_preset


def _init_view_v3():
    """v3 viewport: show the part colours (material colours; 「灰模」 in the panel still works) and no per-object outline — in Solid
    mode Blender draws a dark outline around every object, i.e. also along each part border, which looked like a seam."""
    for win in bpy.context.window_manager.windows:
        if not any(a.type == 'VIEW_3D' for a in win.screen.areas):
            continue
        for a in win.screen.areas:
            if a.type == 'VIEW_3D':
                a.spaces.active.shading.show_object_outline = False
        with bpy.context.temp_override(window=win):
            win.scene.cs.shade = "MAT"               # keeps the panel's 「显示」 choice in step
        PN.apply_shading(win.screen, "MAT")
    return None


def after_boot():
    """call AFTER s1_panel.boot(): v3 sub-panels (their parent CS_PT_main must exist) + 「之前」= v020"""
    props = _props()
    _REG[:] = [props] + OPS + PANELS
    for cl in _REG:
        bpy.utils.register_class(cl)
    bpy.types.Scene.cs_v3 = PointerProperty(type=props)
    obj = PN.body()
    if obj is None:
        return
    if "v3_before_state" in obj:
        S.before = json.loads(obj["v3_before_state"])
        S.before_pts = dict(off=json.loads(obj.get("v3_before_off", "{}")), mirror=True)
        S.before_name = "v020（v3 之前）"
    if PT.ensure_display(obj):                   # files saved before the seamless-border display get it here (display only)
        PN.log("零件显示已升级：零件与身体在显示平滑之后才分开，交界处不再有明暗台阶（只影响显示，顶点不变）")
    if not bpy.app.background:
        bpy.app.timers.register(_init_view_v3, first_interval=1.2)     # after s1_panel's own view init (0.8 s)
    PN.log("v3 已加载：胖瘦/体型在「v3 胖瘦/体型」，胸臀零件在「v3 胸/臀零件」；零件安装点在绿点面板")


def unregister():
    for cl in reversed(_REG):
        try:
            bpy.utils.unregister_class(cl)
        except RuntimeError:
            pass
    if hasattr(bpy.types.Scene, "cs_v3"):
        del bpy.types.Scene.cs_v3
