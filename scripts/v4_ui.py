"""Small v4 entry panel. Reuses existing drag tool, history, save and visibility operators."""
import bpy
from bpy.props import EnumProperty
import s1_panel as PN,a_points as A
class CS_OT_v4_focus(bpy.types.Operator):
    bl_idname='cs.v4_focus'
    bl_label='切换点系统'
    kind:EnumProperty(items=[('GREEN','绿点','结构/安装'),('WHITE','白点','身体表面'),('RED','红点','胸臀造型')])
    def execute(self,context):
        A.P.show=True;A.P.show_green=self.kind=='GREEN';A.P.show_white=self.kind=='WHITE';A.P.show_shape=self.kind=='RED'
        groups={'GREEN':['中轴','肩臂','下肢','骨盆与轮廓','胸部','臀部','零件安装'],
                'WHITE':['白·肩腋','白·背','白·腰腹','白·手臂','白·腿','白·脚','白·胸','白·臀'],
                'RED':['红·胸','红·臀']}
        A.P.groups_open=set(groups[self.kind]);A.P.selected=None
        obj=PN.body();obj['v4_point_focus']=self.kind;obj.show_wire=self.kind=='WHITE';obj.show_all_edges=self.kind=='WHITE'
        if obj.get('v5_interaction'):
            import v3_parts
            for part in v3_parts.part_objects().values():
                if part:part.show_wire=self.kind=='WHITE';part.show_all_edges=self.kind=='WHITE'
        for area in context.screen.areas if context.screen else []:
            if area.type=='VIEW_3D':area.tag_redraw()
        return {'FINISHED'}
class CS_PT_v4(bpy.types.Panel):
    bl_label='v4 先选要调什么'
    bl_idname='CS_PT_v4'
    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category=PN.TAB
    bl_parent_id='CS_PT_main';bl_order=0
    @classmethod
    def poll(cls,context):
        obj=PN.body();return obj is not None and not obj.get('v5_interaction')
    def draw(self,context):
        l=self.layout;row=l.row(align=True)
        for kind,label in [('GREEN','绿：结构/安装'),('WHITE','白：身体表面'),('RED','红：胸臀造型')]:
            row.operator('cs.v4_focus',text=label).kind=kind
        l.operator('cs.point_tool',text='开启/关闭拖点工具',icon='EMPTY_ARROWS')
        if PN.body().get('v5_interaction'):
            l.label(text='单击选点 → G → X/Y/Z → 数字（厘米）')
            l.label(text='Enter/左键确认；Esc/右键取消；Shift精细移动')
            l.label(text='开启点工具后生效；黄点和名称表示当前选择')
        else:
            l.label(text='拖动单侧默认镜像；Esc取消；松开记一次历史')
        l.label(text='紫色是问题提示，不能拖。白点模式显示网格。')
        l.label(text='完成身体后再确认；面容仍未开始。')
def register():
    for cls in (CS_OT_v4_focus,CS_PT_v4):bpy.utils.register_class(cls)
