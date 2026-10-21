"""Stage A UI: reference images / prompt / presets / green points (draw + drag) / mesh-health panel.
Registered from s1_panel.boot().  All state-changing actions go through s1_panel.commit() so that undo/redo, before/after and the
health check see them.  Honest labels: the prompt entry is a RULE-based vocabulary parser (not an LLM); the free AI entry is not done."""
import os
import json
import time
import types

import bpy
import gpu
import blf
import numpy as np
from bpy.props import StringProperty, EnumProperty, IntProperty, FloatProperty, BoolProperty, PointerProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

import s1_core as C
import a_points as A
import a_inputs as I
import s1_panel as PN

TAB = PN.TAB
S = PN.S
PICK_PX = 14
UI = types.SimpleNamespace(flash=None, preset_items=[("0", "（无）", "")], und=None, job=None, handlers=[], tool=None, last_fit=None)


# ------------------------------------------------------------------ properties
def _preset_items(self, context):
    return UI.preset_items


def refresh_presets():
    items = []
    for i, (n, d, _) in enumerate(I.builtin_presets()):
        items.append(("B%d" % i, n, d))
    for i, (n, d, _) in enumerate(I.user_presets()):
        items.append(("U%d" % i, "用户：" + n, d or "用户保存的预设"))
    UI.preset_items = items or [("none", "（无）", "")]


class CSInProps(bpy.types.PropertyGroup):
    ref_path: StringProperty(name="参考图文件", subtype='FILE_PATH', default="", description="选一张图（正/侧/背/三视图拼图）；图会被复制进项目 reference/inputs，原图不动")
    ref_view: EnumProperty(name="视角", items=[("front", "正面", ""), ("side", "侧面", ""), ("back", "背面", ""), ("sheet", "三视图拼图(自动切成 正/侧/背)", ""),
                                             ("q34", "3/4 面", ""), ("other", "其他", "")], default="front")
    prompt: StringProperty(name="", default="", description="例如：高挑苗条的女性，腰细，胯宽，胸大")
    preset: EnumProperty(name="预设", items=_preset_items)
    preset_name: StringProperty(name="名称", default="我的预设")
    free_ai: StringProperty(name="", default="", description="自由 AI 入口：未完成（本机没有被授权的 AI 环境）")
    dx: FloatProperty(name="X cm", default=0.0, min=-20, max=20, description="角色左 = +X")
    dy: FloatProperty(name="Y cm", default=0.0, min=-20, max=20, description="前 = −Y")
    dz: FloatProperty(name="Z cm", default=0.0, min=-20, max=20)


# ------------------------------------------------------------------ reference images
def _rid_prop():
    return IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})


class CS_OT_ref_add(bpy.types.Operator):
    bl_idname = "cs.ref_add"
    bl_label = "添加参考图"
    bl_description = "把上面选的图复制进项目并显示在模型后面（多张、多视角都可以；拼图会自动切成正/侧/背）"

    def execute(self, context):
        sc = context.scene.cs_in
        path = bpy.path.abspath(sc.ref_path)
        try:
            new = I.add_reference(PN.body(), path, view=sc.ref_view)
        except Exception as ex:
            PN.log("✗ 添加参考图失败：%s" % ex)
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        PN.log("✓ 已添加 %d 张参考图：" % len(new) + "、".join("#%d %s" % (r["id"], r["view"]) for r in new) + "（项目内副本 reference/inputs，原图未动）")
        return {'FINISHED'}


class CS_OT_ref_remove(bpy.types.Operator):
    bl_idname = "cs.ref_remove"
    bl_label = "移除"
    bl_description = "从场景和列表里移除这张参考图（项目里的副本文件保留，不删除任何文件）"
    rid: _rid_prop()

    def execute(self, context):
        I.remove_reference(PN.body(), self.rid)
        PN.log("已移除参考图 #%d（文件仍保留在 reference/inputs）" % self.rid)
        return {'FINISHED'}


class CS_OT_ref_replace(bpy.types.Operator):
    bl_idname = "cs.ref_replace"
    bl_label = "替换"
    bl_description = "用上面选的图替换这张参考图（旧副本保留在项目里）"
    rid: _rid_prop()

    def execute(self, context):
        sc = context.scene.cs_in
        try:
            I.replace_reference(PN.body(), self.rid, bpy.path.abspath(sc.ref_path))
        except Exception as ex:
            PN.log("✗ 替换失败：%s" % ex)
            return {'CANCELLED'}
        PN.log("已替换参考图 #%d（旧副本保留）" % self.rid)
        return {'FINISHED'}


class CS_OT_ref_show(bpy.types.Operator):
    bl_idname = "cs.ref_show"
    bl_label = "显示"
    bl_description = "显示/隐藏这张参考图（rid=0 为全部）"
    rid: _rid_prop()
    on: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        I.set_visible(PN.body(), self.on, None if self.rid == 0 else self.rid)
        return {'FINISHED'}


class CS_OT_ref_tag(bpy.types.Operator):
    bl_idname = "cs.ref_tag"
    bl_label = "改视角标签"
    bl_description = "循环切换这张图的视角标签：正面 → 侧面 → 背面"
    rid: _rid_prop()

    def execute(self, context):
        obj = PN.body()
        cur = next(r["view"] for r in I.refs(obj) if r["id"] == self.rid)
        order = ["front", "side", "back"]
        nxt = order[(order.index(cur) + 1) % 3] if cur in order else "front"
        I.set_view_tag(obj, self.rid, nxt)
        PN.log("参考图 #%d 视角标签 → %s" % (self.rid, nxt))
        return {'FINISHED'}


# ------------------------------------------------------------------ generate from references
def generation_report(res, rows):
    out = ["参考图分析（轮廓法，系统 Python + OpenCV/scipy；没有姿态模型权重）：用到 %d 张图" % len(res.get("views", []))]
    for n in res.get("notes", []):
        if not n.startswith("SHEET_CROPS"):
            out.append("  · " + n)
    errs = sorted(I.fit_rows_report(rows), key=lambda r: -abs(r["err"]))
    out.append("拟合后最大偏差（占身高%）：" + "；".join("%s@%.2f %+.1f" % (r["q"], r["fl"], r["err"] * 100) for r in errs[:4]))
    out.append("限制：只对剪影拟合；绝对尺寸/体重/罩杯/软硬/胸间距从图里看不出来；胸部形状（圆/锥/台阶）剪影看不出，需在绿点/滑块里手调")
    return out


def commit_generated(full, label="参考图生成草模"):
    obj = PN.body()
    A.P.off = {}
    A.save_props(obj)
    PN.commit(full, label)
    PN.sync_props(bpy.context.scene, C.get_state(obj))


def generate_sync(obj):
    """Whole pipeline without UI (used by the modal operator's finish step and by tests): analyse -> fit -> commit."""
    res = I.analyse_references(obj)
    job = I.FitJob(obj, res)
    full, rows, mp = job.run_all()
    UI.last_fit = dict(res=res, rows=I.fit_rows_report(rows), obj0=job.obj0, best=job.best)
    commit_generated(full)
    PN.log(*generation_report(res, rows))
    return full, rows, res


class CS_OT_generate(bpy.types.Operator):
    bl_idname = "cs.generate_refs"
    bl_label = "分析参考图并生成草模"
    bl_description = "对「已添加且可见」的参考图做轮廓分析，再用现有滑块拟合出草模（约 30–90 秒，ESC 取消；结果进入历史，可撤销）"
    _timer = None

    def invoke(self, context, event):
        obj = PN.body()
        if not I.active_refs(obj):
            PN.log("✗ 没有参考图：请先添加（或改用提示词/预设创建）")
            return {'CANCELLED'}
        if S.showing_before:
            self.report({'WARNING'}, "请先切回「现在」")
            return {'CANCELLED'}
        try:
            self.res = I.analyse_references(obj)
        except Exception as ex:
            PN.log("✗ 参考图分析失败：%s" % ex)
            return {'CANCELLED'}
        self.st0, self.pts0 = C.get_state(obj), A.snapshot()
        self.job = I.FitJob(obj, self.res)
        UI.job = self.job
        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        context.window_manager.modal_handler_add(self)
        PN.log("开始拟合…（ESC 取消）")
        return {'RUNNING_MODAL'}

    def _stop(self, context):
        context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)
        UI.job = None

    def modal(self, context, event):
        if event.type == 'ESC':
            self._stop(context)
            obj = PN.body()
            A.restore(obj, self.pts0)
            C.apply_state(obj, self.st0)
            PN.log("已取消拟合，模型恢复原状")
            return {'CANCELLED'}
        if event.type == 'TIMER':
            done = self.job.advance(0.2)
            context.workspace.status_text_set("拟合 %d%%（ESC 取消）" % int(self.job.progress * 100))
            if done:
                full, rows, mp = self.job.finish()
                UI.last_fit = dict(res=self.res, rows=I.fit_rows_report(rows), obj0=self.job.obj0, best=self.job.best)
                self._stop(context)
                commit_generated(full)
                PN.log(*generation_report(self.res, rows))
                return {'FINISHED'}
        return {'RUNNING_MODAL'}


# ------------------------------------------------------------------ prompt / presets / free AI
def apply_prompt(text):
    """understand -> report -> (if anything was recognised) generate a draft from the DEFAULT body + the targets."""
    obj = PN.body()
    und = I.understand(text)
    UI.und = und
    obj["cs_prompt"] = text
    obj["cs_ai_understanding"] = json.dumps(und, ensure_ascii=False)
    lines = [und["mode"]]
    for r in und["recognized"]:
        lines.append("✓ 「%s」→ %s %s" % (r["phrase"], r["label"], json.dumps(r["targets"], ensure_ascii=False)))
    for r in und["later"]:
        lines.append("↷ 「%s」→ %s" % (r["phrase"], r["reason"]))
        PN._todo(r["phrase"], r["reason"])
    for u in und["unrecognized"]:
        lines.append("✗ 「%s」没有认出（词表里没有）；未执行，已记入待办" % u)
        PN._todo(u, "提示词创建：词表未覆盖")
    for c in und["conflicts"]:
        lines.append("⚠ 冲突：" + c)
    for n in und["notes"]:
        lines.append("ℹ " + n)
    if not und["state_targets"]:
        lines.append("没有任何可执行的体型描述，未生成。")
        PN.log(*lines)
        return und, None
    st = I.state_from_understanding(und)
    A.P.off = {}
    A.save_props(obj)
    full = PN.commit(st, "提示词创建：" + text[:24])
    PN.sync_props(bpy.context.scene, C.get_state(obj))
    lines.append("→ 已按上述认出的项生成草模（其余参数为中性默认；可撤销，可继续用绿点/滑块/一句话修改）")
    PN.log(*lines)
    return und, full


class CS_OT_prompt(bpy.types.Operator):
    bl_idname = "cs.prompt_create"
    bl_label = "理解并生成"
    bl_description = "规则指令模式：按词表理解这段描述，列出认出/没认出/属于后续阶段/冲突的内容，并生成草模"

    def execute(self, context):
        if S.showing_before:
            self.report({'WARNING'}, "请先切回「现在」")
            return {'CANCELLED'}
        t = context.scene.cs_in.prompt.strip()
        if not t:
            return {'CANCELLED'}
        apply_prompt(t)
        return {'FINISHED'}


def _preset_data(key):
    if key.startswith("B"):
        n, d, st = I.builtin_presets()[int(key[1:])]
        return n, st, {}, True
    n, d, data = I.user_presets()[int(key[1:])]
    return n, data["state"], data.get("offsets", {}), data.get("mirror", True)


def apply_preset(key):
    obj = PN.body()
    name, st, off, mir = _preset_data(key)
    I.apply_preset_data(obj, st, off, mir)
    full = PN.commit(C.get_state(obj), "预设：" + name)
    PN.sync_props(bpy.context.scene, full)
    PN.log("已应用预设「%s」（可撤销；预设只是起点，仍可用绿点/滑块修改）" % name)
    return full


class CS_OT_preset_apply(bpy.types.Operator):
    bl_idname = "cs.preset_apply"
    bl_label = "应用预设"

    def execute(self, context):
        if S.showing_before:
            self.report({'WARNING'}, "请先切回「现在」")
            return {'CANCELLED'}
        k = context.scene.cs_in.preset
        if k == "none":
            return {'CANCELLED'}
        apply_preset(k)
        return {'FINISHED'}


class CS_OT_preset_save(bpy.types.Operator):
    bl_idname = "cs.preset_save"
    bl_label = "把当前存为预设"
    bl_description = "保存当前滑块 + 绿点偏移 + 镜像状态到 presets/user/（重名自动加序号，不覆盖）"

    def execute(self, context):
        p = I.save_user_preset(PN.body(), context.scene.cs_in.preset_name, "面板保存")
        refresh_presets()
        PN.log("已保存预设：" + os.path.basename(p))
        return {'FINISHED'}


class CS_OT_free_ai(bpy.types.Operator):
    bl_idname = "cs.free_ai"
    bl_label = "提交（未完成）"
    bl_description = "自由 AI 入口：未完成——这里只把请求写进 docs/AI_inbox.md，不会改模型"

    def execute(self, context):
        t = context.scene.cs_in.free_ai.strip()
        if t:
            PN.log("【自由 AI 入口·未完成】" + I.submit_free_ai(t))
        return {'FINISHED'}


# ------------------------------------------------------------------ green points
def _view_vec(rv3d):
    return Vector(rv3d.view_matrix.inverted().col[2].xyz).normalized()


def visible_to_view(obj, rv3d, w, pid=None):
    """see a_points.point_visible (interior points are x-ray; surface points need a clear line of sight or a camera-facing skin)"""
    return A.point_visible(obj, pid, w, _view_vec(rv3d))


def pick(obj, region, rv3d, mouse, max_px=PICK_PX, occlusion=True):
    """nearest visible green point to the mouse (region coordinates) or None"""
    pos = A.positions(obj)
    best = None
    for p in A.P.reg:
        if not A.is_visible(p):
            continue
        s = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(pos[p["id"]]))
        if s is None:
            continue
        d = (s - Vector(mouse)).length
        if d <= max_px and (best is None or d < best[0]):
            if occlusion and not visible_to_view(obj, rv3d, pos[p["id"]], p["id"]):
                continue
            best = (d, p["id"])
    return best[1] if best else None


def _health_points():
    """points within 6 cm of a health issue (drawn purple)"""
    if not S.health or not S.health["issues"]:
        return set()
    obj = PN.body()
    pos = A.positions(obj, C.raw_coords_cached(obj))
    out = set()
    for i in S.health["issues"][:80]:
        q = np.array(i["pos"])
        for pid, v in pos.items():
            if np.linalg.norm(v - q) < 0.06:
                out.add(pid)
    return out


COL_MAIN, COL_SUB, COL_SEL, COL_RED = (0.1, 0.95, 0.25, 1), (0.55, 1.0, 0.65, 1), (1.0, 0.9, 0.1, 1), (0.72, 0.22, 1.0, 1)
# COL_RED = mesh-health marker colour: PURPLE since v019 (red now means the bust / glute shape points)
COL_WMAIN, COL_WSUB = (1.0, 1.0, 1.0, 1), (0.82, 0.86, 0.92, 1)      # stage-B white points (main / sub)
COL_RMAIN, COL_RSUB = (1.0, 0.12, 0.12, 1), (1.0, 0.45, 0.45, 1)      # v019 red shape points (main / sub)
COL_HEALTH = (0.72, 0.22, 1.0, 0.95)                                   # purple flags at the located mesh problems


def _draw_points3d():
    obj = PN.body()
    ctx = bpy.context
    if obj is None or not A.P.reg or not A.P.show or ctx.space_data is None or ctx.space_data.type != 'VIEW_3D':
        return
    rv3d = ctx.region_data
    vd = _view_vec(rv3d)
    pos = A.positions(obj, C.raw_coords_cached(obj))
    red = _health_points() if A.P.show_red else set()
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('ALPHA')
    inner = A.interior_ids(obj)
    for xray in (False, True):
        gpu.state.depth_test_set('NONE' if xray else 'LESS_EQUAL')
        buckets = {COL_MAIN: [], COL_SUB: [], COL_SEL: [], COL_RED: [], COL_WMAIN: [], COL_WSUB: [], COL_RMAIN: [], COL_RSUB: []}
        for p in A.P.reg:
            if not A.is_visible(p) or (p["id"] in inner) != xray:
                continue
            q = Vector(pos[p["id"]]) + (Vector((0, 0, 0)) if xray else vd * 0.02)
            if p.get("red"):
                dflt = COL_RMAIN if p["main"] else COL_RSUB
            elif p.get("white"):
                dflt = COL_WMAIN if p["main"] else COL_WSUB
            else:
                dflt = COL_MAIN if p["main"] else COL_SUB
            key = COL_SEL if p["id"] == A.P.selected else (COL_RED if p["id"] in red and not p.get("red") else dflt)   # red shape points keep their colour (the purple flags mark the spot)
            buckets[key].append(tuple(q))
        for col, size in ((COL_SUB, 7), (COL_WSUB, 7), (COL_RSUB, 7), (COL_MAIN, 9), (COL_WMAIN, 9), (COL_RMAIN, 10), (COL_RED, 11), (COL_SEL, 12)):
            pts = buckets[col]
            if not pts:
                continue
            if col in (COL_WMAIN, COL_WSUB, COL_RMAIN, COL_RSUB):   # dark halo so white / red points stay visible on the light clay body
                gpu.state.point_size_set(size + 5)
                bh = batch_for_shader(sh, 'POINTS', {"pos": pts})
                sh.bind()
                sh.uniform_float("color", (0.05, 0.08, 0.2, 1.0))
                bh.draw(sh)
            gpu.state.point_size_set(size)
            b = batch_for_shader(sh, 'POINTS', {"pos": pts})
            sh.bind()
            sh.uniform_float("color", col)
            b.draw(sh)
    if S.health and A.P.show_red:                                # purple flags at the located problems (always visible)
        gpu.state.depth_test_set('NONE')
        pts = [tuple(i["pos"]) for i in S.health["issues"][:80] if i["kind"] != "ground"]
        if pts:
            gpu.state.point_size_set(6)
            b = batch_for_shader(sh, 'POINTS', {"pos": pts})
            sh.bind()
            sh.uniform_float("color", COL_HEALTH)
            b.draw(sh)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')
    if obj.get('v5_interaction'):
        import v5_interaction
        v5_interaction.draw_axes(obj)


def _draw_label2d():
    obj = PN.body()
    ctx = bpy.context
    if obj is not None and obj.get('v5_interaction'):
        import v5_interaction
        v5_interaction.draw_status(obj)
    fl = getattr(UI, "flash", None)
    if fl and ctx.space_data is not None and ctx.space_data.type == 'VIEW_3D' and time.time() - fl[2] < 10.0:
        blf.size(0, 16)
        y = ctx.region.height - 120
        blf.color(0, 1.0, 0.95, 0.2, 1.0)
        blf.position(0, 80, y, 0)
        blf.draw(0, "拖点：" + fl[0])
        blf.size(0, 14)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        for m in (fl[1] or ["已移动"]):
            for ln in PN._wrap(m, 34):
                y -= 22
                blf.position(0, 80, y, 0)
                blf.draw(0, ln)
    if obj is None or A.P.selected is None or ctx.space_data is None or ctx.space_data.type != 'VIEW_3D':
        return
    sp = A.P.by_id.get(A.P.selected)
    if sp is None:
        return
    s = view3d_utils.location_3d_to_region_2d(ctx.region, ctx.region_data, Vector(A.pos(obj, sp["id"], C.raw_coords_cached(obj))))
    if s is None:
        return
    blf.size(0, 15)
    blf.color(0, 1.0, 0.95, 0.2, 1.0)
    blf.position(0, s.x + 12, s.y + 8, 0)
    blf.draw(0, "%s%s" % (sp["label"], "（占位·只记录）" if sp["record_only"] else ""))


def install_draw():
    if bpy.app.background or UI.handlers:
        return
    UI.handlers = [bpy.types.SpaceView3D.draw_handler_add(_draw_points3d, (), 'WINDOW', 'POST_VIEW'),
                   bpy.types.SpaceView3D.draw_handler_add(_draw_label2d, (), 'WINDOW', 'POST_PIXEL')]


def remove_draw():
    for h in UI.handlers:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
        except Exception:
            pass
    UI.handlers = []


def _tag_redraw():
    wm = bpy.context.window_manager
    for w in wm.windows:
        for a in w.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


def finish_drag(obj, drag, label_extra=""):
    """one drag gesture = ONE history entry"""
    sp = drag.sp
    st = C.get_state(obj)
    PN.commit(st, "拖点：%s%s" % (sp["label"], label_extra))
    PN.sync_props(bpy.context.scene, C.get_state(obj))


class CS_OT_point_tool(bpy.types.Operator):
    bl_idname = "cs.point_tool"
    bl_label = "三色点编辑工具"
    bl_description = "开启后：在视图里按住绿点拖动（垂直于视线的平面）；松开 = 一条历史记录；拖动中 Esc / 右键取消。再点一次按钮关闭"

    def invoke(self, context, event):
        if S.tool_on:
            S.tool_on = False
            return {'FINISHED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            return {'CANCELLED'}
        self.area = context.area
        self.win = next(r for r in self.area.regions if r.type == 'WINDOW')
        self.drag = None
        self.last = 0.0
        self.pending = None
        S.tool_on = True
        A.P.show = True
        install_draw()
        context.window_manager.modal_handler_add(self)
        PN.log("三色点工具已开启：单击选点后 G 移动，也能直接拖动" if PN.body().get('v5_interaction') else "绿点工具已开启：在视图里按住点拖动")
        return {'RUNNING_MODAL'}

    def _inside(self, event):
        r = self.win
        if not (r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height):
            return False
        for ur in self.area.regions:
            if ur.type in ('UI', 'TOOLS', 'HEADER', 'TOOL_HEADER') and ur.width > 1 and \
                    ur.x <= event.mouse_x < ur.x + ur.width and ur.y <= event.mouse_y < ur.y + ur.height:
                return False
        return True

    def _target(self, event):
        mouse = (event.mouse_x - self.win.x, event.mouse_y - self.win.y)
        rv3d = self.area.spaces.active.region_3d
        return view3d_utils.region_2d_to_location_3d(self.win, rv3d, mouse, Vector(self.drag.p0))

    def _smooth_off(self, obj):
        """v019: display smoothing (subdivision) off while dragging - re-subdividing the whole body on every mouse move was most of the lag"""
        self.smooth_was = C.is_display_smooth(obj)
        if self.smooth_was:
            C.set_display_smooth(obj, False)

    def _smooth_back(self, obj):
        if getattr(self, "smooth_was", False):
            C.set_display_smooth(obj, True)
        self.smooth_was = False

    def _preview(self, event):
        t0 = time.time()
        msgs = self.drag.preview(np.array(self._target(event)))
        self.last = time.time()
        self.cost = self.last - t0
        S.n_preview = getattr(S, "n_preview", 0) + 1
        self.msgs = msgs
        UI.flash = (self.drag.sp["label"], list(msgs), time.time())
        _tag_redraw()

    def _apply(self, event, guard=True):
        t0 = time.time()
        st, msgs = self.drag.update(np.array(self._target(event)), guard=guard)
        self.last = time.time()
        self.cost = self.last - t0
        self.msgs = msgs
        UI.flash = (self.drag.sp["label"], list(msgs), time.time())
        _tag_redraw()

    def modal(self, context, event):
        if PN.body() is not None and PN.body().get('v5_interaction'):
            import v5_interaction
            return v5_interaction.modal(self, context, event)
        if not S.tool_on:
            if self.drag:
                self.drag.cancel()
                self._smooth_back(PN.body())
            _tag_redraw()
            PN.log("绿点工具已关闭")
            return {'FINISHED'}
        obj = PN.body()
        if self.drag is None:
            if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self._inside(event) and not S.showing_before:
                rv3d = self.area.spaces.active.region_3d
                pid = pick(obj, self.win, rv3d, (event.mouse_x - self.win.x, event.mouse_y - self.win.y))
                if pid:
                    A.P.selected = pid
                    self.drag = A.Drag(obj, pid)
                    self._smooth_off(obj)
                    self.m0 = PN.measure() if (PN.stage_of(obj) == "B" and not pid.startswith("w_")) else None
                    self.msgs = []
                    _tag_redraw()
                    return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            self.pending = event
            if time.time() - self.last > max(0.03, 1.5 * getattr(self, "cost", 0.0)):       # fast linear preview (~ms); slow updates throttle themselves
                self._preview(event)
            return {'RUNNING_MODAL'}
        if (event.type == 'ESC' or event.type == 'RIGHTMOUSE') and event.value == 'PRESS':
            self.drag.cancel()
            self._smooth_back(obj)
            PN.log("已取消这次拖动（模型恢复）")
            self.drag = None
            _tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self._apply(event)
            self._smooth_back(obj)
            d, self.drag = self.drag, None
            finish_drag(obj, d)
            if getattr(self, "m0", None) is not None:
                m1 = PN.measure()
                ch = ["%s %+.1f %s" % (k, (m1[k] - self.m0[k]) * (100 if k == "身高" else 100), "cm" if k == "身高" else "%H")
                      for k in self.m0 if abs(m1[k] - self.m0[k]) > (0.005 if k == "身高" else 0.004)]
                if ch:
                    self.msgs = list(self.msgs) + ["【注意】绿点是大形点，会带动整个身体：" + "、".join(ch) + "。阶段A已确认的大比例被改了；不想要请点撤销，细节改动请用白点"]
            self.m0 = None
            PN.log("拖点 " + d.sp["label"] + "：" + ("；".join(self.msgs) if self.msgs else "完成"))
            UI.flash = (d.sp["label"], list(self.msgs), time.time())
            _tag_redraw()
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}


class CS_OT_pt_select(bpy.types.Operator):
    bl_idname = "cs.pt_select"
    bl_label = "选点"
    pid: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        A.P.selected = self.pid
        _tag_redraw()
        return {'FINISHED'}


class CS_OT_pt_group(bpy.types.Operator):
    bl_idname = "cs.pt_group"
    bl_label = "展开/收起"
    group: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        (A.P.groups_open.discard if self.group in A.P.groups_open else A.P.groups_open.add)(self.group)
        _tag_redraw()
        return {'FINISHED'}


class CS_OT_pt_show(bpy.types.Operator):
    bl_idname = "cs.pt_show"
    bl_label = "显示点"
    kind: EnumProperty(items=[("ALL", "全部", ""), ("GREEN", "绿点", ""), ("WHITE", "白点", ""), ("RED", "紫点（网格问题标记）", ""), ("SHAPE", "红点（胸臀造型）", "")],
                       default="ALL", options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        if self.kind == "GREEN":
            A.P.show_green = not A.P.show_green
        elif self.kind == "WHITE":
            A.P.show_white = not A.P.show_white
        elif self.kind == "RED":
            A.P.show_red = not A.P.show_red
        elif self.kind == "SHAPE":
            A.P.show_shape = not A.P.show_shape
        else:
            A.P.show = not A.P.show
        _tag_redraw()
        return {'FINISHED'}


class CS_OT_pt_mirror(bpy.types.Operator):
    bl_idname = "cs.pt_mirror"
    bl_label = "镜像"
    mode: EnumProperty(items=[("OFF", "关", ""), ("ON", "开", ""), ("L", "以左为准", ""), ("R", "以右为准", ""), ("AVG", "取平均", "")],
                       options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        if S.showing_before:
            return {'CANCELLED'}
        obj = PN.body()
        ok, msg = A.set_mirror(obj, self.mode != "OFF", None if self.mode in ("ON", "OFF") else self.mode)
        if ok:
            PN.commit(C.get_state(obj), "镜像：" + ("开" if self.mode != "OFF" else "关"))
        PN.log(("✓ " if ok else "⚠ ") + msg)
        return {'FINISHED'}


class CS_OT_pt_reset(bpy.types.Operator):
    bl_idname = "cs.pt_reset"
    bl_label = "复位"
    bl_description = "复位：选中的点（及其镜像）或全部点的局部偏移，并把该点负责的滑块放回草模值（可撤销）"
    mode: EnumProperty(items=[("SEL", "选中点", ""), ("ALL", "全部点", "")], options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        if S.showing_before:
            return {'CANCELLED'}
        obj = PN.body()
        if self.mode == "SEL" and not A.P.selected:
            PN.log("没有选中的点")
            return {'CANCELLED'}
        st = A.reset_points(obj, None if self.mode == "ALL" else [A.P.selected])
        PN.commit(st, "复位点：" + ("全部" if self.mode == "ALL" else A.P.by_id[A.P.selected]["label"]))
        PN.sync_props(context.scene, C.get_state(obj))
        PN.log("已复位（可撤销）")
        return {'FINISHED'}


class CS_OT_pt_nudge(bpy.types.Operator):
    bl_idname = "cs.pt_nudge"
    bl_label = "按数值移动选中点"
    bl_description = "和拖动同一套算法（含镜像/中线/地面/占位保护），按上面 X/Y/Z cm 数值移动选中的点，一条历史记录。左 = +X，前 = −Y"

    def execute(self, context):
        if S.showing_before:
            return {'CANCELLED'}
        sc = context.scene.cs_in
        obj = PN.body()
        if not A.P.selected:
            PN.log("先选一个点")
            return {'CANCELLED'}
        d = A.Drag(obj, A.P.selected)
        st, msgs = d.update(d.p0 + np.array([sc.dx, sc.dy, sc.dz]) / 100.0)
        finish_drag(obj, d, "（数值）")
        PN.log("移动 %s：%s" % (d.sp["label"], "；".join(msgs) if msgs else "完成"))
        return {'FINISHED'}


# ------------------------------------------------------------------ health
class CS_OT_health_check(bpy.types.Operator):
    bl_idname = "cs.health_check"
    bl_label = "重新检查网格"

    def execute(self, context):
        import a_health as H
        S.health_prev = S.health
        S.health = H.check(PN.body())
        PN.log(*H.summarize(S.health, 5))
        _tag_redraw()
        return {'FINISHED'}


class CS_OT_health_goto(bpy.types.Operator):
    bl_idname = "cs.health_goto"
    bl_label = "定位"
    idx: IntProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        if not S.health or self.idx >= len(S.health["issues"]):
            return {'CANCELLED'}
        i = S.health["issues"][self.idx]
        r3d = context.area.spaces.active.region_3d
        r3d.view_location = Vector(i["pos"])
        r3d.view_distance = 0.9
        _tag_redraw()
        return {'FINISHED'}


# ------------------------------------------------------------------ panels
class _Sub:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = TAB
    bl_parent_id = "CS_PT_main"


class CS_PT_inputs(_Sub, bpy.types.Panel):
    bl_label = "输入与创建（参考图 / 提示词 / 预设）"
    bl_idname = "CS_PT_inputs"
    bl_order = 0

    def draw(self, context):
        lay = self.layout
        obj = PN.body()
        sc = context.scene.cs_in
        box = lay.box()
        box.label(text="① 参考图（可多张；可不放图）")
        box.prop(sc, "ref_path", text="")
        box.prop(sc, "ref_view", text="")
        box.operator("cs.ref_add", icon='IMAGE_DATA')
        act = I.active_refs(obj)
        if act:
            row = box.row(align=True)
            row.operator("cs.ref_show", text="全部显示").rid = 0
            o = row.operator("cs.ref_show", text="全部隐藏")
            o.rid, o.on = 0, False
        for r in act:
            row = box.row(align=True)
            row.operator("cs.ref_tag", text="#%d %s" % (r["id"], {"front": "正", "side": "侧", "back": "背"}.get(r["view"], r["view"]))).rid = r["id"]
            e = bpy.data.objects.get("CS_Ref_%d" % r["id"])
            shown = e is not None and not e.hide_viewport
            o = row.operator("cs.ref_show", text="", icon='HIDE_OFF' if shown else 'HIDE_ON')
            o.rid, o.on = r["id"], (not shown)
            row.operator("cs.ref_replace", text="换").rid = r["id"]
            row.operator("cs.ref_remove", text="", icon='X').rid = r["id"]
        if act:
            box.operator("cs.generate_refs", icon='MOD_ARMATURE')
            if UI.job is not None:
                box.label(text="拟合中 %d%%" % int(UI.job.progress * 100))
        box = lay.box()
        box.label(text="② 提示词（规则指令模式·词表解析，不是大模型）")
        box.prop(sc, "prompt")
        box.operator("cs.prompt_create", icon='TEXT')
        if UI.und:
            u = UI.und
            box.label(text="认出 %d / 没认出 %d / 后续阶段 %d / 冲突 %d" % (len(u["recognized"]), len(u["unrecognized"]), len(u["later"]), len(u["conflicts"])))
        box = lay.box()
        box.label(text="③ 无图预设（可保存/命名/再应用）")
        box.prop(sc, "preset", text="")
        box.operator("cs.preset_apply", icon='IMPORT')
        box.prop(sc, "preset_name")
        box.operator("cs.preset_save", icon='EXPORT')
        box = lay.box()
        box.label(text="④ 自由 AI 入口：未完成")
        for ln in PN._wrap(I.AI_STATUS, 22):
            box.label(text=ln)
        box.prop(sc, "free_ai")
        box.operator("cs.free_ai")


class CS_PT_points(_Sub, bpy.types.Panel):
    bl_label = "绿点 / 白点 / 红点（拖动改体型）"
    bl_idname = "CS_PT_points"
    bl_order = 1

    def draw(self, context):
        lay = self.layout
        obj = PN.body()
        sc = context.scene.cs_in
        row = lay.row(align=True)
        row.operator("cs.point_tool", text="拖动工具：%s" % ("开（点此关闭）" if S.tool_on else "关（点此开启）"), icon='RESTRICT_SELECT_OFF', depress=S.tool_on)
        row = lay.row(align=True)
        row.operator("cs.pt_show", text="全部点：%s" % ("开" if A.P.show else "关"), icon='HIDE_OFF' if A.P.show else 'HIDE_ON').kind = "ALL"
        row = lay.row(align=True)
        row.operator("cs.pt_show", text="绿点（大形）：%s" % ("显示" if A.P.show_green else "隐藏"), depress=A.P.show_green).kind = "GREEN"
        row.operator("cs.pt_show", text="白点（局部）：%s" % ("显示" if A.P.show_white else "隐藏"), depress=A.P.show_white).kind = "WHITE"
        row = lay.row(align=True)
        row.operator("cs.pt_show", text="红点（胸臀造型）：%s" % ("显示" if A.P.show_shape else "隐藏"), depress=A.P.show_shape).kind = "SHAPE"
        row.operator("cs.pt_show", text="紫点（网格问题）：%s" % ("显示" if A.P.show_red else "隐藏"), depress=A.P.show_red).kind = "RED"
        lay.label(text="绿=大比例　白=体态/局部　红=胸臀造型")
        lay.label(text="紫点不能拖：标出网格上有尖点/折痕的位置")
        box = lay.box()
        asym = A.asymmetry(obj)
        if A.P.mirror:
            box.operator("cs.pt_mirror", text="左右镜像：开（点此关闭）", icon='MOD_MIRROR').mode = "OFF"
        else:
            box.label(text="左右镜像：关" + ("（两侧不对称 %.1f cm）" % (asym * 100) if asym > 1e-4 else ""))
            if asym > 1e-4:
                row = box.row(align=True)
                for m, t in (("L", "以左为准"), ("R", "以右为准"), ("AVG", "取平均")):
                    row.operator("cs.pt_mirror", text=t).mode = m
            else:
                box.operator("cs.pt_mirror", text="开启镜像").mode = "ON"
        sp = A.P.by_id.get(A.P.selected) if A.P.selected else None
        box = lay.box()
        if sp is None:
            box.label(text="没有选中的点（在视图里点绿/白/红点，或下面列表里选）")
        else:
            box.label(text="选中：%s（%s）" % (sp["label"], sp["group"]))
            for ln in PN._wrap(sp["hint"], 20):
                box.label(text=ln)
            p = A.pos(obj, sp["id"], C.raw_coords_cached(obj))
            o = A.P.off.get(sp["id"], [0, 0, 0])
            box.label(text="位置 %.3f %.3f %.3f m" % tuple(p))
            box.label(text="局部偏移 %.1f cm　可动轴 %s" % (np.linalg.norm(o) * 100, sp["axes"]))
            if sp["record_only"]:
                box.label(text="占位点：只记录，不改模型", icon='INFO')
            if sp.get("red"):
                st_now = C.get_state(obj)
                for key, ids in sp["owners"].items():
                    for cid in ids:
                        box.label(text="%s拖 → %s = %+.2f" % ({"X": "左右", "Y": "前后", "Z": "上下"}.get(key, key), C.CTRL[cid]["label"], st_now.get(cid, 0.0)))
                box.label(text="红点只改造型，不改绿点比例/白点体态", icon='INFO')
            row = box.row(align=True)
            row.prop(sc, "dx")
            row.prop(sc, "dy")
            row.prop(sc, "dz")
            box.operator("cs.pt_nudge", icon='EMPTY_ARROWS')
            box.operator("cs.pt_reset", text="复位选中点").mode = "SEL"
        lay.operator("cs.pt_reset", text="复位全部点").mode = "ALL"
        for g in A.GROUPS:
            ps = [p for p in A.P.reg if p["group"] == g]
            if not ps:
                continue
            op = g in A.P.groups_open
            row = lay.row(align=True)
            row.operator("cs.pt_group", text="%s（%d 点%s）" % (g, len(ps), "" if op else ("，仅主点显示" if any(p["main"] for p in ps) else "，默认隐藏")), icon='TRIA_DOWN' if op else 'TRIA_RIGHT', emboss=False).group = g
            if op:
                col = lay.column(align=True)
                for p in ps:
                    col.operator("cs.pt_select", text=("● " if p["main"] else "○ ") + p["label"], depress=(p["id"] == A.P.selected)).pid = p["id"]


class CS_PT_health(_Sub, bpy.types.Panel):
    bl_label = "网格健康检查"
    bl_idname = "CS_PT_health"
    bl_order = 2

    def draw(self, context):
        import a_health as H
        lay = self.layout
        h = S.health
        lay.operator("cs.health_check", icon='VIEWZOOM')
        if h is None:
            lay.label(text="还没有检查结果")
            return
        new = H.new_since(S.health_prev, h) if S.health_prev is not None else []
        if new:
            b = lay.box()
            b.alert = True
            b.label(text="⚠ 这一步新增 %d 处问题" % len(new), icon='ERROR')
            b.operator("cs.undo", text="撤销这一步", icon='LOOP_BACK')
        if not h["issues"]:
            lay.label(text="未发现尖刺 / 折角 / 翻面 / 腿越线 / 离地", icon='CHECKMARK')
            return
        lay.label(text="共 %d 处：%s" % (h["n"], "、".join("%s %d" % (k, v) for k, v in h["by_kind"].items())))
        for k, i in enumerate(h["issues"][:10]):
            row = lay.row(align=True)
            row.operator("cs.health_goto", text="%s@%s %.0f%s" % (i["kind"], i["region"], i["value"], i["unit"])).idx = k
        lay.label(text="紫色 = 问题提示；红点 = 胸臀造型")


CLASSES = [CSInProps, CS_OT_ref_add, CS_OT_ref_remove, CS_OT_ref_replace, CS_OT_ref_show, CS_OT_ref_tag, CS_OT_generate, CS_OT_prompt,
           CS_OT_preset_apply, CS_OT_preset_save, CS_OT_free_ai, CS_OT_point_tool, CS_OT_pt_select, CS_OT_pt_group, CS_OT_pt_show,
           CS_OT_pt_mirror, CS_OT_pt_reset, CS_OT_pt_nudge, CS_OT_health_check, CS_OT_health_goto, CS_PT_inputs, CS_PT_points, CS_PT_health]


def register():
    refresh_presets()
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.cs_in = PointerProperty(type=CSInProps)
    install_draw()
    obj = PN.body()
    if obj is not None:
        if "cs_ref_metrics" in obj and I.active_refs(obj):
            try:
                C.REF_OVERRIDE = I.ref_override(json.loads(obj["cs_ref_metrics"]))
            except Exception:
                pass
        try:
            I.rebuild_empties(obj)
        except Exception as ex:
            print("reference empties:", ex)


def unregister():
    remove_draw()
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass
    if hasattr(bpy.types.Scene, "cs_in"):
        del bpy.types.Scene.cs_in


def inputs_record(obj):
    """what the draft was built from (task book §8 'input files / state files')"""
    return dict(references=I.refs(obj), prompt=obj.get("cs_prompt"), understanding=json.loads(obj["cs_ai_understanding"]) if "cs_ai_understanding" in obj else None,
                reference_metrics=json.loads(obj["cs_ref_metrics"]) if "cs_ref_metrics" in obj else None,
                last_fit=UI.last_fit and dict(rows=UI.last_fit["rows"], objective_start=UI.last_fit["obj0"], objective_end=UI.last_fit["best"]),
                mode_label=I.MODE_LABEL, free_ai=I.AI_STATUS, saved=time.strftime("%Y-%m-%d %H:%M:%S"))
