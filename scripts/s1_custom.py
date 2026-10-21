"""Stage-1 TRIAL deformations that MPFB does not ship ("新增·试验·未入库").

Design: every custom deformation is a function of the CURRENT mixed coordinates (MPFB sliders already applied)
and returns a per-vertex displacement.  The sum of all displacements is stored in ONE extra shape key "cs_custom"
(value 1.0) that sits on top of MPFB's keys, so it follows every slider move (recomputed by apply_state) and
"cs_custom" = 0 gives back pure MPFB.  Nothing here touches the confirmed library; values are bounded 0..1.

Controls
  legs_together  并腿站姿：两腿绕髋关节向内收（MPFB 默认腿分开约 8°，参考图是并腿）。这是姿势类形变。
  glute_proj     臀后凸：只在臀区（分区“臀”）沿向后/略向外推出，边缘平滑衰减，保护腰/大腿/胯前。
  knees_in       膝并拢（v002）：参考图两膝几乎相碰、脚踝仍分开；只在膝附近把两腿向中线平移，髋与脚踝不动。
  knees_back     膝后移（v002）：参考图侧面膝盖比脚踝-髋连线更靠后（约 1.5 %身高）；只在膝附近整腿向后平移。
  glute_low      下臀丰满/臀峰下移（v002）：参考图臀最凸点比 MPFB 低、下臀更满；下臀向后加厚、臀上部略收。
  bust_round     胸形圆润（v004）：MPFB 的胸偏锥形，参考图是圆润的半球/水滴；整个胸丘向前隆起（乳头处最多，边缘衰减到 0），配合柔化更圆。
  bust_upper     上胸饱满（v004）：MPFB 锁骨下到胸之间是凹台阶，参考图上半球一直满到锁骨下；该区域向前填充。
  bust_lower     胸下缘圆润（v005）：MPFB 胸下缘是一道硬台阶（折棱 + 多边形轮廓）；对下胸缘一带做限幅拉普拉斯平滑，
                 把台阶抹成柔和的下垂弧线（下缘略上收、体积略减，由拟合补偿）。v004 试过 Taubin 柔化（bust_soften，最大位移仅 2~10 mm，台阶纹丝不动），已删除。
"""
import math
import os
import json
import hashlib
import numpy as np
import bpy

import s1_core as C

KEY = "cs_custom"
MAX_ADDUCT_DEG = 8.0        # legs_together = 1.0  -> each leg rotated 8 deg toward the midline (measured from the reference)
GLUTE_MAX_M = 0.03          # glute_proj = 1.0     -> 3 cm extra projection at the glute centre (~1.8 % of body height)
KNEE_IN_M = 0.03            # knees_in = 1.0       -> 3 cm toward the midline at knee height (measured gap error ~2 cm)
KNEE_BACK_M = 0.03          # knees_back = 1.0     -> 3 cm backwards at knee height (measured knee offset ~2.5 cm)
# glute_low = 1.0: extra back projection (in % of body height) by height (fraction of H above the sole);
# from the v001 residual table (lower glute too shallow, peak too high / too far back)
GLUTE_LOW_PROFILE = [(0.40, 0.0), (0.45, 0.5), (0.49, 0.6), (0.51, 1.0), (0.53, 0.4), (0.55, -0.3), (0.57, -0.55),
                     (0.59, -0.4), (0.62, 0.0)]

BUST_R = 0.11               # bust radius around the nipple (m); the MPFB bust ellipsoid zone is ~0.105
BUST_ROUND_M = 0.02         # bust_round = 1.0  -> 2 cm extra forward at the nipple, fading to 0 at the bust radius
BUST_UPPER_M = 0.025        # bust_upper = 1.0  -> 2.5 cm forward fill between clavicle and nipple
LOWER_ITERS = 25            # bust_lower: Laplacian passes (lambda 0.5) on the finished surface, lower-bust band only
LOWER_CLAMP_M = 0.018       # bust_lower = 1.0  -> smoothing displacement capped at 1.8 cm (stops sparse-mesh outliers)
LOWER_BAND = (-0.03, -0.105)  # band in nipple-relative height (m): starts just under the nipple, ends below the fold
LOWER_HALF_W = 0.115        # band half width (m) around the nipple

# v008 additions (diagnosed in renders/A_v008_diag): the nipple "spike" (Laplacian excess 8-9 mm) and the bust-root
# hard fold come from the MAXED MPFB cupsize cone on a sparse mesh - isolation showed cupsize 1.0->0.65 removes all 4
# spikes, zeroing the three trial deformers does NOT.  bust_smooth is a clamped Laplacian over the WHOLE bust cap
# (apex + cleavage + the ribcage ring at the root) that flattens the tip spike and softens the root fold WITHOUT losing
# the smooth mound (low-frequency geometry is untouched; only the high-curvature outliers are pulled in).
BUST_SMOOTH_ITERS = 26
BUST_SMOOTH_CLAMP_M = 0.017  # bust_smooth = 1.0 -> per-vertex smoothing capped at 1.7 cm (kills the 8-9 mm tip spike)
# v009: the v008 whole-cap clamped bust_smooth left a residual triangle around the nipple (clamp stops at the threshold,
# not at "visually smooth") AND shrank the mound (it smooths low-frequency curvature too).  Replaced by a DEDICATED,
# LOCAL nipple/areola suppressor (nipple_flat) that leaves the mound volume to cupsize + bust_round.  nipple_flat is a
# Dirichlet "harmonic fill": inside a small disc on each breast the surface is relaxed to the smooth patch spanning the
# disc RIM (the rim sits on the mound and is pinned), so the MPFB nipple cone + the sparse-mesh areola triangle are
# erased while the forward projection / volume (set by the rim) is preserved - no cupsize/firmness/threshold tricks.
NIP_R = 0.050               # disc radius around each nipple (m); covers the areola triangle, stays well inside the mound
NIP_FILL_ITERS = 80         # harmonic (Dirichlet) relaxation passes inside the disc, rim fixed
NIP_FLAT_CLAMP_M = 0.05     # safety cap (the fill only ever lowers the small nipple bump; cap only catches outliers)
# glute_smooth: clamped Laplacian over the glute silhouette - rounds the boxy outer edge, the horizontal lower fold and
# the abrupt side peak seen in the side/back renders.  v009: widened from "back-facing only" to the whole BACK+LATERAL
# buttock plus a band across the glute-thigh fold and up into the hip (the box corner lives at the lateral edge where
# y ~= pelvis-centre, which the v008 back-only mask skipped), and the clamp raised 1.4->3.0 cm so the hard box corner
# actually rounds.  Low-curvature geometry (the peak projection, 胯宽) has ~0 Laplacian so it barely moves.
GLUTE_SMOOTH_ITERS = 26
GLUTE_SMOOTH_CLAMP_M = 0.030
GLUTE_CURV_LO = 0.004       # below this local curvature (m) a vertex is "smooth" (peak / hip width) and is not moved
GLUTE_CURV_HI = 0.011       # at/above this it is a hard edge (box corner / lower fold) and gets the full smoothing
# back_slim (ORIGINAL - restored verbatim from work/codex_v010_baseline_20261001/s1_custom.py so v007-v010 reopen
# byte-identically; the v013 back fix lives in the NEW control id `back_anatomy` below, never by editing this one).
# pull the UPPER-BACK + neck/trapezius back surface (+y) forward, fading to 0 at the torso centre so the bust
# (front, -y) is never touched.  Addresses the "后背过厚 / 颈肩沉重" look.  It does NOT chase the side_back reference above
# 0.6 H, which is polluted by the reference's hanging hair (static ref 0.022 vs dynamic 0.0412 at 0.625 H - see docs).
BACK_SLIM_M = 0.028         # back_slim = 1.0 -> up to 2.8 cm of depth removed from the upper back, tapering to the sides

# back_anatomy (v013 NEW control, does NOT touch back_slim): fix the user-reported upper-back "too WIDE + flat PLATE"
# WITHOUT the baseline-pollution of editing back_slim and WITHOUT the MPFB torso-muscle-dorsi target (which leaked onto the
# bust front and front waist, measured 4.6 / 3.1 mm in v012, and WIDENED the ribcage +6.7 mm).  It does three things, each
# confined to a BACK-ONLY skin mask (upback_w x back-facing, 0 on the bust/front-waist/arms/head):
#   1) LOW-FREQUENCY depth slim - pull forward by a HEAVILY smoothed depth field so a muscle peak and the neighbouring
#      spine groove shift together => the slab thins but the high-frequency relief (scapula / erector spinae) survives.
#   2) POSTERIOR LATERAL width correction - a smooth, spatially-smoothed inward shift of the far-lateral back that fades to
#      0 at/above the acromion (shoulder-peak width kept) => the erroneous upper-back width is REDUCED, never increased.
#   3) a shallow midline SPINE FURROW.
# back_anatomy composes WITH the original back_slim (both are computed on the MPFB base co0 and summed): back_slim supplies
# v010's clean depth slim (peaks pulled in = thin), back_anatomy's UNSHARP term pushes the high-frequency relief back out so
# the result is thin AND not a flat plate.  back_anatomy therefore carries NO depth term of its own (avoids the thickness
# regression and the uniform-shift folds); it only restores relief + a spine furrow + a gentle posterior width correction.
BACK_ANATOMY_RELIEF = 1.1      # unsharp gain on (co_y - lowpass): re-establish the scapula / erector-spinae relief the depth
#                                slim compresses (peaks out, grooves in); low-frequency bulk untouched.
BACK_ANATOMY_LOWPASS = 30      # Laplacian passes for the low-pass used by the unsharp relief term
BACK_ANATOMY_TAPER_M = 0.018   # posterior lateral correction; smooth transition, shoulder peaks and chest preserved
BACK_ANATOMY_GROOVE_M = 0.009  # shallow spine furrow at the midline (m)
BACK_ANATOMY_FIELD_SMOOTH = 2  # Laplacian passes on the displacement FIELD (fold-free at mask edges)

CUSTOM = [
    dict(id="legs_together", label="并腿站姿（两腿内收）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="姿势类：MPFB 没有此控制，绕髋关节内收，最大 8°"),
    dict(id="glute_proj", label="臀后凸（独立于体积）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="只作用于臀区，最大 3 cm，边缘平滑衰减"),
    dict(id="knees_in", label="膝并拢（膝部内收）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="姿势类：膝附近两腿向中线平移，最大 3 cm；髋和脚踝不动"),
    dict(id="knees_back", label="膝后移（腿线微后弓）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="姿势类：膝附近整条腿向后平移，最大 3 cm；髋和脚踝不动"),
    dict(id="glute_low", label="下臀丰满 / 臀峰下移", group="试验·新增", kind="custom", lo=0.0, hi=1.6,
         default=0.0, have="新增·试验·未入库", stage="①试", note="下臀向后加厚、臀上部略收；1.0 ≈ 峰值 1% 身高，滑块最大 1.6"),
    dict(id="bust_round", label="胸形圆润（胸丘隆起）", group="试验·新增", kind="custom", lo=0.0, hi=1.6,
         default=0.0, have="新增·试验·未入库", stage="①试", note="整个胸丘向前隆起，乳头处最多、边缘衰减到 0；1.0 = +2 cm"),
    dict(id="bust_upper", label="上胸饱满（锁骨下填充）", group="试验·新增", kind="custom", lo=0.0, hi=1.6,
         default=0.0, have="新增·试验·未入库", stage="①试", note="锁骨下到乳头之间向前填充；1.0 = 最大 +2.5 cm"),
    dict(id="bust_lower", label="胸下缘圆润（抹平硬台阶）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="下胸缘带限幅平滑，最大 1.8 cm；乳头及上胸不动；下缘略上收、体积略减"),
    dict(id="bust_smooth", label="胸面圆化（消乳尖刺/乳根折）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="v008：对整片胸帽（尖→乳根）限幅拉普拉斯，消乳尖尖刺与乳根硬折；大形不变，最大 1.6 cm"),
    dict(id="nipple_flat", label="乳头/乳晕几何抑制（局部谐波填充·已弃用）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="v009：谐波填充=极小曲面，在凸胸丘上会向内塌成凹坑（用户目视否决），v010 起弃用，改 nipple_dome"),
    dict(id="nipple_dome", label="乳头凸面重建（球面拟合）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="v010：乳头圆盘外侧支撑环拟合球面(正曲率)，用凸面重写盘内深度，消乳锥/乳晕三角且中心前凸≥环，无凹坑无尖头"),
    dict(id="glute_smooth", label="臀面圆化（臀峰/臀下缘）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="v008：对臀面限幅平滑，圆化偏方的臀峰上缘与盒状臀下缘；最大 1.4 cm"),
    dict(id="back_slim", label="上背/颈肩减厚", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试", note="v008：上背+颈肩后缘向前收，前胸不动；不贴被头发污染的后缘轮廓；最大 2.8 cm"),
    dict(id="back_anatomy", label="上背修形（收窄+脊柱沟，保起伏）", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·试验·未入库", stage="①试",
         note="v013：仅后背蒙皮——低频收薄保起伏+后侧收窄(0 at 肩峰)+脊柱沟；不碰前胸/前腰/臂/头；替代会泄漏的 back_dorsi"),
    dict(id="glute_round", label="旧试验·臀加肉（本版未采用）", group="试验·新增", kind="custom", lo=0.0, hi=1.4,
         default=0.0, have="新增·试验·未入库", stage="①试",
         note="兼容v013试验：会增加臀部尺寸，不满足保尺寸修形；v014使用参考01臀部结构修正，不启用此项。"),
    dict(id="glute_anatomy_ref01", label="参考01·臀部结构修正", group="试验·新增", kind="custom", lo=0.0, hi=1.0,
         default=0.0, have="新增·参考01专用·待验收", stage="①试",
         note="采用MPFB现有圆润臀形，局部融合后恢复v010臀宽和最大后突；保护胸和前腰。固定拓扑的参考01形变，不是通用解剖推理。"),
    dict(id="bust_sag", label="胸 上托↔下垂", group="阶段B·胸", kind="custom", lo=-1.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="整个胸丘整体上移(-)或下移(+)，最大 1.5 cm；乳根处衰减到 0，不动胸廓"),
    dict(id="bust_yaw", label="胸 朝向（内收↔外展）", group="阶段B·胸", kind="custom", lo=-1.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="胸丘整体向外(+)或向内(-)平移，最大 1.2 cm，左右镜像"),
    dict(id="bust_pitch", label="胸 朝向（朝下↔朝上）", group="阶段B·胸", kind="custom", lo=-1.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="上极后退、下极前凸=朝上(+)；反之朝下(-)，最大 1.5 cm"),
    dict(id="bust_lowfull", label="胸 下半球饱满度", group="阶段B·胸", kind="custom", lo=-1.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="乳头以下向前加厚(+)或收薄(-)，最大 2 cm；上半球请用“上胸饱满”"),
    dict(id="glute_up", label="臀峰高度（下↔上）", group="阶段B·臀", kind="custom", lo=-1.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="臀后部整体上提(+)或下移(-)，最大 1.5 cm；臀宽和后突量不改"),
    dict(id="glute_leg", label="臀↔腿过渡柔化", group="阶段B·臀", kind="custom", lo=0.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="臀下缘到大腿后侧一带限幅平滑，最大 1.5 cm"),
    dict(id="armpit_relax", label="腋下抚平（腋前/腋窝/腋后折痕）", group="阶段B·细化", kind="custom", lo=0.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="v019：只在两侧腋下（腋前褶、腋窝、腋后褶）做限幅平滑，最大 1.2 cm；其他部位不动，比例尺寸不变"),
    dict(id="back_relax", label="背部抚平（腋后折线/肩胛上沿横纹）", group="阶段B·细化", kind="custom", lo=0.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="v020：腋后往下沿肩胛外缘那条斜折线 + 斜方肌/肩胛上部的横向波纹做限幅平滑，最大 1.2 cm；比例尺寸不变"),
    dict(id="arm_raise", label="抬臂（0=A字站姿，1=T字双臂水平）", group="阶段B·姿势", kind="custom", lo=0.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="v019：整条手臂绕肩关节向上转（1 = 手臂水平，约 47°），肩/腋下渐变过渡；随时拉回 0 = 原样；只是姿势，不改体型数据"),
    dict(id="refine_auto", label="AI 自动细化（只修尖点/折痕）", group="阶段B·细化", kind="custom", lo=0.0, hi=1.0, default=0.0, have="新增·阶段B", stage="B",
         note="只对“比中性体型多出的尖点/折角”做限幅平滑，最大 1 cm；头/手/脚不动；平滑处位移≈0"),
] + [dict(id=i, label=lb, group=g, kind="custom", lo=-1.0, hi=1.0, default=0.0, have="新增·阶段B·红点", stage="B", note=nt) for i, lb, g, nt in (
    ("r_bust_upper", "上胸坡（凹·水滴 ↔ 饱满·圆）", "红·胸造型", "v019 红点：乳头以上到锁骨下之间的坡面，负=内凹（水滴形），正=饱满（圆形）；实测最大约 1 cm；乳头不动"),
    ("r_bust_lower", "下胸弧（扁 ↔ 圆）", "红·胸造型", "v019 红点：乳头到乳下线之间的弧，正=更圆更兜，负=更扁；实测最大约 0.9 cm；乳头和乳下线不动"),
    ("r_nip_pitch", "乳头朝向（朝下 ↔ 朝上）", "红·胸造型", "v019 红点：只动乳头周围约 6 cm 的胸尖，正=朝上，负=朝下；实测最大约 1.1 cm；胸根不动"),
    ("r_nip_yaw", "乳头朝向（朝内 ↔ 朝外）", "红·胸造型", "v019 红点：只动胸尖，正=朝外，负=朝内；实测最大约 0.9 cm；胸根不动"),
    ("r_imf", "乳下线高度（低 ↔ 高）", "红·胸造型", "v019 红点：乳房下缘那条线上移（正）或下移（负），实测最大只有约 0.3 cm（再大会在乳下线起折痕）；乳头不动"),
    ("r_cleavage", "乳沟间距（近 ↔ 远）", "红·胸造型", "v019 红点：两胸内侧面向外（正=乳沟变宽）或向内（负=更聚拢），实测最大约 0.5 cm；胸骨中线不动"),
    ("r_side_bust", "侧胸（收 ↔ 丰满）", "红·胸造型", "v019 红点：胸外侧（靠腋下那一侧）向外前方鼓出（正）或收进（负），实测最大约 0.4 cm"),
    ("r_glute_shelf", "上臀（平 ↔ 翘·饱满）", "红·臀造型", "v019 红点：臀峰以上到腰之间（上臀/腰臀过渡）向后鼓（正=翘）或收平（负），实测最大约 1.3 cm"),
    ("r_glute_lowfull", "下臀（瘪 ↔ 饱满）", "红·臀造型", "v019 红点：臀下线以上那一段（下臀）向后鼓（正）或收（负），实测最大约 1.5 cm；臀后凸测量线不动"),
    ("r_glute_fold_d", "臀下线深浅（浅 ↔ 深）", "红·臀造型", "v019 红点：臀和大腿交界那条线压深（正）或变浅（负），实测最大约 1 cm"),
    ("r_glute_fold_h", "臀下线高度（低 ↔ 高）", "红·臀造型", "v019 红点：臀下线整体上移（正=更提）或下移（负），实测最大约 1.3 cm"),
    ("r_glute_hollow", "侧臀凹陷（填平 ↔ 凹）", "红·臀造型", "v019 红点：臀外侧偏后（大转子后方）凹进（正）或填平（负），实测最大约 1 cm；胯宽测量处不动"),
)]
CUSTOM_IDS = [c["id"] for c in CUSTOM]

# NOTE: the v011/v012 `back_dorsi` MPFB bi-control (torso-muscle-dorsi) has been REMOVED: the raw target moved the bust
# front and the front waist (measured 4.6 / 3.1 mm against the v010 actual coords) and widened the ribcage, violating the
# bust/front-waist protection.  v013 restores the back relief geometrically on a back-only mask via `back_anatomy` instead.

_CTX = {}


def register():
    for c in CUSTOM:
        if c["id"] not in C.CTRL:
            C.CONTROLS.append(c)
            C.CTRL[c["id"]] = c
    C.CUSTOM_HOOK = apply_custom


def _addat(acc, idx, vals):
    """fast equivalent of np.add.at(acc, idx, vals) (acc 1-D or (n,k); vals scalar or matching rows)"""
    n = acc.shape[0]
    if acc.ndim == 1:
        w = None if np.isscalar(vals) else np.asarray(vals, dtype=np.float64)
        acc += np.bincount(idx, weights=w if w is not None else np.full(len(idx), float(vals)), minlength=n)
    else:
        v = np.asarray(vals, dtype=np.float64)
        for k in range(acc.shape[1]):
            acc[:, k] += np.bincount(idx, weights=v[:, k], minlength=n)



_NB = {}


def _nsum(x, ea, eb):
    """per-vertex sum of the neighbours' values over the edge list (ea, eb): same result as two np.add.at calls, via a cached sorted adjacency"""
    c = _NB.get((id(ea), id(eb)))
    if c is None or c[0] is not ea or c[1] is not eb:
        src, dst = np.concatenate([ea, eb]), np.concatenate([eb, ea])
        o = np.argsort(src, kind="stable")
        ss = src[o]
        starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        c = (ea, eb, dst[o], starts, ss[starts])
        _NB[(id(ea), id(eb))] = c
    out = np.zeros(x.shape, dtype=np.float64)
    out[c[4]] = np.add.reduceat(x[c[2]], c[3], axis=0)
    return out


def _smooth(field, ea, eb, iters):
    f = field.astype(np.float64).copy()
    n = len(f)
    cnt = np.ones(n)
    _addat(cnt, ea, 1)
    _addat(cnt, eb, 1)
    for _ in range(iters):
        f = (f + _nsum(f, ea, eb)) / cnt
    return f


def _ctx(obj):
    key = (obj.name, len(obj.data.vertices))
    c = _CTX.get(key)
    if c is not None:
        return c
    zone = C.load_zones(obj)
    if zone is None:
        raise RuntimeError("basemesh has no zone_id attribute (run compute_zones/store_zones first)")
    ed = C._adjacency(obj)
    counts = np.asarray([p.loop_total for p in obj.data.polygons], dtype=np.int32)
    indices = np.empty(len(obj.data.loops), dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", indices)
    topology_hash = hashlib.sha256(counts.tobytes() + indices.tobytes()).hexdigest()
    ea, eb = ed[:, 0], ed[:, 1]
    leg = np.isin(zone, [C.ZID[k] for k in ("thigh", "calf", "foot", "hip", "glute")]).astype(np.float64)
    glute = (zone == C.ZID["glute"]).astype(np.float64)
    torso = np.isin(zone, [C.ZID[k] for k in ("ribcage", "bust", "waist")]).astype(np.float64)
    bust = (zone == C.ZID["bust"]).astype(np.float64)
    upback = np.isin(zone, [C.ZID[k] for k in ("ribcage", "neck")]).astype(np.float64)
    c = dict(zone=zone, ea=ea, eb=eb, topology_sha256=topology_hash, obj_name=obj.name,
             torso_w=np.clip(_smooth(torso, ea, eb, 3), 0.0, 1.0),
             bust_w=np.clip(_smooth(bust, ea, eb, 4), 0.0, 1.0),
             bust_w2=np.clip(_smooth(bust, ea, eb, 8), 0.0, 1.0),
             upback_w=np.clip(_smooth(upback, ea, eb, 4), 0.0, 1.0),
             nip_ids=C.group_indices(obj, "nipple"),
             shoulder_ids=C.group_indices(obj, "joint-l-shoulder"),
             leg_w=np.clip(_smooth(leg, ea, eb, 4), 0.0, 1.0),
             glute_w=np.clip(_smooth(glute, ea, eb, 6), 0.0, 1.0),
             glute_w2=np.clip(_smooth(glute, ea, eb, 12), 0.0, 1.0),   # v009: wider band - bleeds across the glute-thigh fold and up into the hip for silhouette rounding
             body=C.group_indices(obj, "body"),
             hip_ids=C.group_indices(obj, "joint-l-upper-leg"),
             ankle_ids=C.group_indices(obj, "joint-r-ankle"))
    _CTX[key] = c
    return c


def _crotch_z(co, ctx):
    z = co[:, 2]
    body = ctx["body"]
    mid = np.zeros(len(co), bool)
    mid[body] = True
    mid &= (np.abs(co[:, 0]) < 0.012) & (z > 0.35 * z[body].max()) & (z < 0.62 * z[body].max())
    return float(z[mid].min())


def d_legs_together(co, ctx, amount):
    """Rotate each leg about its hip joint toward the midline.  Below the ankle the rotation is replaced by the
    translation the ankle undergoes, so the soles stay flat on the floor."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    P = co[ctx["hip_ids"]]
    hip = P[P[:, 0] > 0].mean(axis=0) if (P[:, 0] > 0).any() else P.mean(axis=0)
    cz = _crotch_z(co, ctx)
    zp = cz + 0.25 * (hip[2] - cz)
    phi = math.radians(MAX_ADDUCT_DEG) * amount
    c, s = math.cos(phi), math.sin(phi)
    x, z = co[:, 0], co[:, 2]
    sg = np.where(x >= 0, 1.0, -1.0)
    dx = np.abs(x) - abs(hip[0])
    dz = z - zp
    ankle_z = float(co[ctx["ankle_ids"], 2].mean())
    dz_a = ankle_z + 0.02 - zp                                   # ankle plane relative to the pivot
    foot = ctx["zone"] == C.ZID["foot"]
    dx_a = float((np.abs(x[foot]) - abs(hip[0])).mean())
    b = np.clip((dz_a - dz) / 0.04, 0.0, 1.0)                    # 0 above the ankle plane .. 1 well below it
    dz_c = np.maximum(dz, dz_a)
    dx_c = dx * (1.0 - b) + dx_a * b
    nx = dx_c * c + dz_c * s                                     # lower points move toward the midline
    nz = -dx_c * s + dz_c * c
    wz = np.clip(1.0 - np.maximum(dz, 0.0) / 0.12, 0.0, 1.0) ** 2   # fade out above the pivot
    wside = np.tanh(np.abs(x) / 0.03)                            # the midline (crotch / cleft) stays put
    w = ctx["leg_w"] * wz * wside
    # keep the soles on the floor: cancel the vertical drop the ankle would get, ramped in from the pivot down
    dz_f = -dx_a * s + dz_a * (c - 1.0)
    ramp = np.clip(dz / dz_a, 0.0, 1.0)
    d[:, 0] = sg * (nx - dx_c) * w
    d[:, 2] = (nz - dz_c) * w - dz_f * ramp * w
    return d


def d_glute_proj(co, ctx, amount):
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    w = ctx["glute_w"]
    P = co[ctx["hip_ids"]]
    yc = float(P[:, 1].mean())                             # hip-joint depth = pelvis centre
    rx, ry = co[:, 0], co[:, 1] - yc
    n = np.sqrt((0.5 * rx) ** 2 + ry ** 2) + 1e-9
    back = (ry > 0).astype(np.float64)                     # only back-facing surface moves
    d[:, 0] = GLUTE_MAX_M * amount * w * back * (0.5 * rx) / n
    d[:, 1] = GLUTE_MAX_M * amount * w * back * ry / n
    return d


def _leg_t(co, ctx):
    """0 at the ankle joint .. 1 at the crotch (fraction of the leg length), clipped."""
    z = co[:, 2]
    z_a = float(co[ctx["ankle_ids"], 2].mean())
    return np.clip((z - z_a) / (_crotch_z(co, ctx) - z_a), 0.0, 1.0)


def d_knees_in(co, ctx, amount):
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    t = _leg_t(co, ctx)
    p = np.clip(np.sin(math.pi * t ** 0.8), 0.0, None) ** 1.4       # 0 at ankle and crotch, peak around the knee
    sg = np.where(co[:, 0] >= 0, 1.0, -1.0)
    wside = np.tanh(np.abs(co[:, 0]) / 0.03)                        # the midline (crotch / cleft) stays put
    d[:, 0] = -sg * KNEE_IN_M * amount * p * ctx["leg_w"] * wside
    return d


def d_knees_back(co, ctx, amount):
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    t = _leg_t(co, ctx)
    p = np.clip(np.sin(math.pi * t), 0.0, None) ** 4                # narrow bump: thigh/calf follow the knee gradually
    d[:, 1] = KNEE_BACK_M * amount * p * ctx["leg_w"]               # +Y = back
    return d


def d_glute_low(co, ctx, amount):
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    body = ctx["body"]
    zmin = float(co[:, 2].min())
    H = float(co[body, 2].max()) - zmin
    fl = (co[:, 2] - zmin) / H
    fx, fy = zip(*GLUTE_LOW_PROFILE)
    amp = np.interp(fl, fx, fy) / 100.0 * H                         # metres at amount = 1
    P = co[ctx["hip_ids"]]
    yc = float(P[:, 1].mean())
    rx, ry = co[:, 0], co[:, 1] - yc
    n = np.sqrt((0.5 * rx) ** 2 + ry ** 2) + 1e-9
    back = (ry > 0).astype(np.float64)
    k = amount * amp * ctx["glute_w"] * back
    d[:, 0] = k * (0.5 * rx) / n
    d[:, 1] = k * ry / n
    return d


def _nipples(co, ctx):
    """(cx, cy, cz) of the +x and -x nipple, read from the current MPFB-only geometry."""
    P = co[ctx["nip_ids"]]
    L, R = P[P[:, 0] > 0], P[P[:, 0] < 0]
    l = L.mean(axis=0) if len(L) else P.mean(axis=0)
    r = R.mean(axis=0) if len(R) else np.array([-l[0], l[1], l[2]])
    return l, r


def _bust_frame(co, ctx):
    """Per-vertex nipple centre (its own side), front-facing weight, and lateral/vertical offsets."""
    l, r = _nipples(co, ctx)
    left = co[:, 0] >= 0
    c = np.where(left[:, None], l[None, :], r[None, :])
    dx, dz = co[:, 0] - c[:, 0], co[:, 2] - c[:, 2]
    wy = np.clip((c[:, 1] + 0.12 - co[:, 1]) / 0.05, 0.0, 1.0)      # front surface only (-y = front); back of the ribcage stays
    return c, dx, dz, wy


def d_bust_round(co, ctx, amount):
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    c, dx, dz, wy = _bust_frame(co, ctx)
    s = np.clip(np.sqrt(dx ** 2 + dz ** 2) / BUST_R, 0.0, 1.0)
    s = np.maximum(s, 0.22)                                         # apex plateau: the inner disc all rides forward by the
    # same amount (flat round top) instead of concentrating the peak on the single sparse nipple vertex -> no tip "nub"
    prof = (1.0 - s * s) ** 2                                       # whole mound rides forward, 0 + flat slope at the base: no ring / crater around the nipple
    d[:, 1] = -BUST_ROUND_M * amount * prof * wy * ctx["bust_w"]
    return d


def d_bust_upper(co, ctx, amount):
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    c, dx, dz, wy = _bust_frame(co, ctx)
    shoulder_z = float(co[ctx["shoulder_ids"], 2].mean())
    t = np.clip(dz / max(shoulder_z - float(c[:, 2].mean()), 0.05), 0.0, 1.0)      # 0 at the nipple .. 1 at the shoulder joint
    pv = np.sin(math.pi * np.clip(t / 0.95, 0.0, 1.0)) ** 1.0
    ax = np.abs(co[:, 0])
    px = np.clip((ax - 0.01) / 0.04, 0.0, 1.0) * (1.0 - np.clip((ax - 0.11) / 0.07, 0.0, 1.0))
    d[:, 1] = -BUST_UPPER_M * amount * pv * px * wy * ctx["torso_w"]
    return d


def _laplace(co, ea, eb, iters, lam=0.5):
    n = len(co)
    cnt = np.zeros(n)
    _addat(cnt, ea, 1.0)
    _addat(cnt, eb, 1.0)
    cnt = np.maximum(cnt, 1.0)
    x = co.astype(np.float64).copy()
    for _ in range(iters):
        x += lam * (_nsum(x, ea, eb) / cnt[:, None] - x)
    return x


def d_bust_lower(co, ctx, amount):
    """POST deformer: `co` is the surface AFTER all the additive deformers (MPFB + round + upper ...)."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    c, dx, dz, wy = _bust_frame(co, ctx)
    band = ((dz < LOWER_BAND[0]) & (dz > LOWER_BAND[1]) & (np.abs(dx) < LOWER_HALF_W)).astype(np.float64) * wy
    w = np.clip(_smooth(band, ctx["ea"], ctx["eb"], 3), 0.0, 1.0)
    e = (_laplace(co, ctx["ea"], ctx["eb"], LOWER_ITERS) - co) * w[:, None]
    m = np.linalg.norm(e, axis=1)
    e *= np.minimum(1.0, LOWER_CLAMP_M / np.maximum(m, 1e-9))[:, None]
    return e * min(amount, 1.0)


def d_bust_smooth(co, ctx, amount):
    """POST deformer: clamped Laplacian over the whole bust cap (apex + cleavage + root ring).  `co` is the surface
    AFTER every additive deformer.  Removes the sparse-mesh tip spike and softens the bust-root fold; the clamp keeps the
    smooth mound (only high-curvature outliers move by up to the clamp)."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    c, dx, dz, wy = _bust_frame(co, ctx)
    w = np.clip(ctx["bust_w2"] * wy, 0.0, 1.0)
    e = (_laplace(co, ctx["ea"], ctx["eb"], BUST_SMOOTH_ITERS) - co) * w[:, None]
    m = np.linalg.norm(e, axis=1)
    e *= np.minimum(1.0, BUST_SMOOTH_CLAMP_M / np.maximum(m, 1e-9))[:, None]
    return e * min(amount, 1.0)


def _harmonic_fill(co, ea, eb, interior, iters):
    """Relax ONLY the `interior` vertices to the average of their neighbours (Dirichlet: every other vertex stays put),
    so the interior converges to the smooth patch spanning the fixed rim just outside it.  Used to erase the nipple cone
    / areola triangle without moving the surrounding mound (the rim is on the mound, so the projection is kept)."""
    x = co.astype(np.float64).copy()
    n = len(co)
    cnt = np.zeros(n)
    _addat(cnt, ea, 1.0)
    _addat(cnt, eb, 1.0)
    cnt = np.maximum(cnt, 1.0)
    idx = np.where(interior)[0]
    if not len(idx):
        return x
    for _ in range(iters):
        nb = _nsum(x, ea, eb) / cnt[:, None]
        x[idx] = nb[idx]
    return x


def d_nipple_flat(co, ctx, amount):
    """POST deformer: erase the MPFB nipple cone + the sparse-mesh areola 'triangle' with a LOCAL harmonic fill inside a
    small disc on each breast.  `co` is the finished additive surface.  The disc rim sits on the mound and is pinned, so
    the forward projection / volume (cupsize + bust_round) is preserved - only the pointed nipple geometry is removed."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    c, dx, dz, wy = _bust_frame(co, ctx)
    r = np.sqrt(dx ** 2 + dz ** 2)
    interior = (r < NIP_R) & (wy > 0.5) & (ctx["bust_w"] > 0.15)     # the disc, front-facing, inside the bust zone
    blend = (np.clip(1.0 - r / NIP_R, 0.0, 1.0) ** 0.5) * wy          # soft weight -> seamless at the rim
    x = _harmonic_fill(co, ctx["ea"], ctx["eb"], interior, NIP_FILL_ITERS)
    e = (x - co) * blend[:, None]
    m = np.linalg.norm(e, axis=1)
    e *= np.minimum(1.0, NIP_FLAT_CLAMP_M / np.maximum(m, 1e-9))[:, None]
    return e * min(amount, 1.0)


# v010: nipple_flat (harmonic fill) is RETIRED - it produced a central crater on every breast (a Dirichlet/membrane
# solution spanning a convex rim is a MINIMAL surface, which sags inward; it can never rebuild a convex dome - confirmed
# by the user on renders/A_v009_deliver).  nipple_dome rebuilds the nipple patch as a CONVEX surface instead: fit a
# sphere (guaranteed positive curvature) to a clean SUPPORT RING of the mound just outside the nipple, then overwrite the
# inner disc's depth (y) with that convex surface, blended to 0 at the rim.  The sphere apex is strictly in front of its
# rim, so the centre projects MORE forward than the surrounding ring (measured in a_exp3 / the build): no tip spike, no
# crater - a continuous convex mound.  Front is -y.
NIP_DOME_R = 0.045           # inner disc that gets rebuilt (m) - covers the cone + areola triangle
NIP_DOME_R2 = 0.095          # outer radius of the support ring the sphere is fit to (clean mound, outside the cone)
NIP_DOME_CLAMP_M = 0.045     # safety cap on the per-vertex depth change


def _fit_sphere(P, w):
    """Weighted algebraic sphere fit.  Returns (centre(3), R) minimizing |p|^2 = 2 c.p + (R^2-|c|^2)."""
    A = np.concatenate([2.0 * P, np.ones((len(P), 1))], axis=1)       # [2x, 2y, 2z, 1]
    b = (P ** 2).sum(axis=1)
    W = w[:, None]
    sol, *_ = np.linalg.lstsq((A * W).T @ A, (A * W).T @ b, rcond=None)
    c = sol[:3]
    R2 = sol[3] + float(c @ c)
    return c, math.sqrt(max(R2, 1e-9))


def d_nipple_dome(co, ctx, amount):
    """POST deformer: rebuild each nipple disc as a CONVEX spherical cap fit from the surrounding mound ring, removing the
    MPFB cone / areola triangle while keeping (indeed guaranteeing) a forward-convex centre.  `co` is the finished
    additive surface.  Only the depth (y) of the inner disc moves; the rim is untouched (blend -> 0)."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    c, dx, dz, wy = _bust_frame(co, ctx)
    r = np.sqrt(dx ** 2 + dz ** 2)
    front = wy > 0.5
    bz = ctx["bust_w"] > 0.12
    for side in (1.0, -1.0):
        sm = (co[:, 0] * side >= 0)
        interior = sm & front & bz & (r < NIP_DOME_R)
        support = sm & front & bz & (r >= NIP_DOME_R) & (r < NIP_DOME_R2)
        if interior.sum() < 3 or support.sum() < 8:
            continue
        sw = np.clip(1.0 - (r[support] - NIP_DOME_R) / (NIP_DOME_R2 - NIP_DOME_R), 0.08, 1.0)  # inner ring weighted more -> seamless rim
        cen, R = _fit_sphere(co[support], sw)
        ii = np.where(interior)[0]
        disc = R ** 2 - (co[ii, 0] - cen[0]) ** 2 - (co[ii, 2] - cen[2]) ** 2
        ty = cen[1] - np.sqrt(np.maximum(disc, 0.0))                  # FRONT (-y) sheet of the sphere -> convex, apex forward
        blend = np.clip(1.0 - r[ii] / NIP_DOME_R, 0.0, 1.0) ** 0.5
        d[ii, 1] += (ty - co[ii, 1]) * blend
    m = np.abs(d[:, 1])
    d[:, 1] *= np.minimum(1.0, NIP_DOME_CLAMP_M / np.maximum(m, 1e-9))
    return d * min(amount, 1.0)


def _curvature(co, ea, eb):
    """per-vertex |co - mean(neighbours)| (one step): large at sharp corners / hard folds, ~0 on smooth bulges."""
    n = len(co)
    cnt = np.zeros(n)
    _addat(cnt, ea, 1.0)
    _addat(cnt, eb, 1.0)
    acc = _nsum(co, ea, eb)
    cnt = np.maximum(cnt, 1.0)
    return np.linalg.norm(co - acc / cnt[:, None], axis=1)


def d_glute_smooth(co, ctx, amount):
    """POST deformer: CURVATURE-TARGETED clamped Laplacian over the BACK+LATERAL buttock plus a band across the
    glute-thigh fold and up into the hip.  Only vertices that are actually sharp (high local curvature - the boxy outer
    corner and the hard horizontal lower fold) are pulled in; the smooth peak bulge and the lateral hip width have ~0
    curvature so they are left alone.  This rounds the box / fold WITHOUT flattening the 臀峰 or narrowing 胯宽.  The box
    corner lives at the LATERAL edge (y ~= pelvis centre), which the v008 back-only mask skipped, so the region is widened
    to include it.  `co` is the finished additive surface."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    P = co[ctx["hip_ids"]]
    yc = float(P[:, 1].mean())                                 # pelvis centre depth
    back = np.clip((co[:, 1] - yc) / 0.05 + 0.5, 0.0, 1.0)     # front fades to 0, lateral (y~=yc) 0.5, full behind
    curv = _curvature(co, ctx["ea"], ctx["eb"])
    cw = np.clip((curv - GLUTE_CURV_LO) / (GLUTE_CURV_HI - GLUTE_CURV_LO), 0.0, 1.0)  # 0 on smooth bulge .. 1 on a hard edge
    cw = np.clip(_smooth(cw, ctx["ea"], ctx["eb"], 2), 0.0, 1.0)                       # grow the edge mask by a ring (no seam)
    w = np.clip(ctx["glute_w2"] * back * cw, 0.0, 1.0)
    e = (_laplace(co, ctx["ea"], ctx["eb"], GLUTE_SMOOTH_ITERS) - co) * w[:, None]
    m = np.linalg.norm(e, axis=1)
    e *= np.minimum(1.0, GLUTE_SMOOTH_CLAMP_M / np.maximum(m, 1e-9))[:, None]
    return e * min(amount, 1.0)


def d_back_slim(co, ctx, amount):
    """ORIGINAL (restored).  Pull the upper-back + neck/trapezius back surface (+y) forward, fading to 0 at the torso centre
    (so the bust, which is at -y, never moves).  Does not chase the hair-polluted side_back reference above 0.6 H."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    rib = ctx["zone"] == C.ZID["ribcage"]
    y0 = float(np.median(co[rib, 1])) if rib.any() else float(co[:, 1].mean())
    back = np.clip((co[:, 1] - y0) / 0.04, 0.0, 1.0)          # 0 at/ahead of the torso centre .. 1 well behind it
    d[:, 1] = -BACK_SLIM_M * amount * back * ctx["upback_w"]
    return d


def d_back_anatomy(co, ctx, amount):
    """v013 NEW: fix the upper-back "too WIDE + flat PLATE" on a BACK-ONLY mask (no bust/front-waist/arm/head motion, no
    MPFB-target leak).  (1) low-frequency depth slim preserving relief; (2) smooth posterior-lateral width reduction that is
    0 at/above the acromion so the shoulder-peak width is kept and the back only gets NARROWER; (3) shallow spine furrow."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    rib = ctx["zone"] == C.ZID["ribcage"]
    y0 = float(np.median(co[rib, 1])) if rib.any() else float(co[:, 1].mean())
    sh_z = float(co[ctx["shoulder_ids"], 2].mean())
    # Use the skeletal shoulder plane rather than a median biased by the separately zoned breast surface.
    y0 = float(co[ctx["shoulder_ids"], 1].mean())
    height = float(np.ptp(co[ctx["body"], 2]))
    def ease(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    ribw = np.clip(_smooth(rib.astype(np.float64), ctx["ea"], ctx["eb"], 5), 0.0, 1.0)
    below_sh = ease((sh_z - 0.025 - co[:, 2]) / 0.065)
    backface = ease((co[:, 1] - y0) / 0.035)
    w = ribw * below_sh * backface
    disp = np.zeros_like(co)
    ax = np.abs(co[:, 0])
    dz = co[:, 2] - (sh_z - 0.065 * height)
    scapula = np.exp(-0.5 * ((ax - 0.047 * height) / (0.025 * height)) ** 2
                    -0.5 * (dz / (0.038 * height)) ** 2)
    latissimus = np.exp(-0.5 * ((ax - 0.060 * height) / (0.030 * height)) ** 2
                       -0.5 * ((co[:, 2] - (sh_z - 0.115 * height)) / (0.052 * height)) ** 2)
    spinal_band = np.exp(-0.5 * ((co[:, 2] - (sh_z - 0.092 * height)) / (0.085 * height)) ** 2)
    spine_groove = np.exp(-0.5 * (co[:, 0] / (0.012 * height)) ** 2) * spinal_band
    erectors = np.exp(-0.5 * ((ax - 0.020 * height) / (0.009 * height)) ** 2) * spinal_band
    # Small modelling landmarks, not exact hidden anatomy recovered from the hair-covered reference.
    disp[:, 1] = (0.0045 * scapula + 0.0025 * latissimus + 0.0015 * erectors - 0.0035 * spine_groove) * w
    disp[:, 0] = -np.sign(co[:, 0]) * BACK_ANATOMY_TAPER_M * ease((ax - 0.070) / 0.110) * w
    disp = np.column_stack([_smooth(disp[:, k], ctx["ea"], ctx["eb"], 2) for k in range(3)])
    protected = ~np.isin(ctx["zone"], [C.ZID["ribcage"], C.ZID["neck"]])
    protected |= (co[:, 1] <= y0) | (co[:, 2] >= sh_z - 0.02)
    disp[protected] = 0.0
    return disp * amount


# glute_round (v013 NEW): a SMOOTH analytic rounded buttock volume that restores the FULL-REGION width + back projection +
# cross-section girth of v010 WITHOUT the triangle/shelf (the old glute_low used a peaked + NEGATIVE-dipped np.interp
# profile -> derivative breaks -> shelf/triangle; its removal in v012 shrank the glute 17 mm and the thigh 10 mm).  This
# pushes the back-facing glute surface out radially (both +y projection and +-x width) with a C-infinity height bump (no
# negative lobe) and keeps the natal cleft (tanh at the midline).  It bleeds into the hip/upper-thigh via glute_w so the
# upper-thigh fullness v010 had is restored (main lower thigh is outside glute_w, so it is untouched).  The amplitudes are
# tuned in a_build_v013 against the v010 ACTUAL saved coords (glute max width >= 390.8 mm, thigh ~ restored).
GLUTE_ROUND_BACK_M = 0.050     # peak backward(+y) projection at amount=1 (m)
GLUTE_ROUND_SIDE_M = 0.034     # peak lateral(+-x) widening at amount=1 (m) -> restores the full-region width
GLUTE_ROUND_PEAK_FL = 0.505    # height of the projection peak (fraction of body height)
GLUTE_ROUND_UP_FL = 0.085      # gaussian half-width above the peak (toward the waist), in fl
GLUTE_ROUND_DN_FL = 0.095      # gaussian half-width below the peak (toward the thigh), in fl -> fuller lower cheek


def d_glute_round(co, ctx, amount):
    """Smooth rounded buttock volume (no dips, no box corner); restores v010 width/projection/girth with a round shape."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    zmin = float(co[ctx["body"], 2].min()); Hh = float(co[ctx["body"], 2].max()) - zmin
    fl = (co[:, 2] - zmin) / Hh
    hz = np.where(fl >= GLUTE_ROUND_PEAK_FL,
                  np.exp(-0.5 * ((fl - GLUTE_ROUND_PEAK_FL) / GLUTE_ROUND_UP_FL) ** 2),
                  np.exp(-0.5 * ((fl - GLUTE_ROUND_PEAK_FL) / GLUTE_ROUND_DN_FL) ** 2))
    P = co[ctx["hip_ids"]]
    yc = float(P[:, 1].mean())
    rx, ry = co[:, 0], co[:, 1] - yc
    back = np.clip((ry - 0.015) / 0.035, 0.0, 1.0)            # strictly back-facing (starts 1.5 cm behind the hip centre) ->
    #                                                           no front-pelvis / front-waist bleed (measured leak removed)
    cleft = np.tanh(np.abs(rx) / 0.020)                        # soft natal cleft kept at the midline
    glonly = (ctx["zone"] == C.ZID["glute"]).astype(np.float64)   # confine the source to the glute zone (+ its own smoothing)
    gw = np.clip(_smooth(glonly, ctx["ea"], ctx["eb"], 5), 0.0, 1.0)
    w = gw * hz * back                                         # glute-zone-confined, bleeds only into the immediate hip/upper-thigh
    d[:, 0] = GLUTE_ROUND_SIDE_M * amount * w * cleft * np.sign(rx)
    d[:, 1] = GLUTE_ROUND_BACK_M * amount * w
    return d


# ------------------------------------------------------------------ stage B deformers
BUST_SAG_M = 0.015
BUST_YAW_M = 0.012
BUST_PITCH_M = 0.015
BUST_LOWFULL_M = 0.020
GLUTE_UP_M = 0.015
GLUTE_LEG_ITERS = 20
GLUTE_LEG_CLAMP_M = 0.015
REFINE_ITERS = 14
REFINE_CLAMP_M = 0.010
REFINE_LO, REFINE_HI = 0.0035, 0.0075       # curvature EXCESS over the neutral body (m): below LO untouched, above HI full smoothing


def _bust_prof(co, ctx):
    c, dx, dz, wy = _bust_frame(co, ctx)
    s = np.clip(np.sqrt(dx ** 2 + dz ** 2) / BUST_R, 0.0, 1.0)
    return c, dx, dz, (1.0 - s * s) ** 2 * wy * ctx["bust_w"]


def d_bust_sag(co, ctx, amount):
    d = np.zeros_like(co)
    if abs(amount) < 1e-9:
        return d
    c, dx, dz, w = _bust_prof(co, ctx)
    d[:, 2] = -BUST_SAG_M * amount * w
    return d


def d_bust_yaw(co, ctx, amount):
    d = np.zeros_like(co)
    if abs(amount) < 1e-9:
        return d
    c, dx, dz, w = _bust_prof(co, ctx)
    d[:, 0] = np.where(co[:, 0] >= 0, 1.0, -1.0) * BUST_YAW_M * amount * w
    return d


def d_bust_pitch(co, ctx, amount):
    d = np.zeros_like(co)
    if abs(amount) < 1e-9:
        return d
    c, dx, dz, w = _bust_prof(co, ctx)
    d[:, 1] = BUST_PITCH_M * amount * np.clip(dz / BUST_R, -1.0, 1.0) * w          # +y = back: upper pole back, lower pole forward = faces up
    return d


def d_bust_lowfull(co, ctx, amount):
    d = np.zeros_like(co)
    if abs(amount) < 1e-9:
        return d
    c, dx, dz, w = _bust_prof(co, ctx)
    low = np.clip(-dz / 0.05, 0.0, 1.0)
    low = low * low * (3.0 - 2.0 * low)                                           # 0 at the nipple line .. 1 below it (smoothstep, no ridge)
    d[:, 1] = -BUST_LOWFULL_M * amount * w * low
    return d


def d_glute_up(co, ctx, amount):
    d = np.zeros_like(co)
    if abs(amount) < 1e-9:
        return d
    yc = float(co[ctx["hip_ids"], 1].mean())
    back = np.clip((co[:, 1] - yc) / 0.05, 0.0, 1.0)
    d[:, 2] = GLUTE_UP_M * amount * ctx["glute_w2"] * back
    return d


def d_glute_leg(co, ctx, amount):
    """POST: clamped Laplacian across the glute-thigh transition (lower buttock border), back-facing only."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    yc = float(co[ctx["hip_ids"], 1].mean())
    back = np.clip((co[:, 1] - yc) / 0.05 + 0.5, 0.0, 1.0)
    zmin = float(co[ctx["body"], 2].min())
    H = float(co[ctx["body"], 2].max()) - zmin
    fl = (co[:, 2] - zmin) / H
    band = np.clip((0.54 - fl) / 0.03, 0.0, 1.0) * np.clip((fl - 0.40) / 0.03, 0.0, 1.0)
    w = np.clip(ctx["glute_w2"] * ctx["leg_w"] * back * band * 1.6, 0.0, 1.0)
    e = (_laplace(co, ctx["ea"], ctx["eb"], GLUTE_LEG_ITERS) - co) * w[:, None]
    m = np.linalg.norm(e, axis=1)
    e *= np.minimum(1.0, GLUTE_LEG_CLAMP_M / np.maximum(m, 1e-9))[:, None]
    return e * min(amount, 1.0)


def _rest_curv(obj, ctx):
    rc = ctx.get("rest_curv")
    if rc is None:
        arr = [obj.get("cs_rest_" + a) for a in "xyz"]
        if any(a is None for a in arr):
            rc = np.zeros(len(obj.data.vertices))
        else:
            rest = np.stack([np.array(list(a), dtype=np.float64) for a in arr], axis=1)
            rc = _curvature(rest, ctx["ea"], ctx["eb"])
        ctx["rest_curv"] = rc
    return rc


def d_refine_auto(co, ctx, amount, obj=None):
    """POST (runs last): AI auto-refine = find where the surface is MORE jagged than the neutral body (curvature excess) and
    pull only those vertices towards their neighbours (clamped).  Smooth areas (excess < REFINE_LO) are not moved, so the
    confirmed big proportions stay; head / hands / feet / helper cubes are protected."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    obj = obj or bpy.data.objects.get(ctx["obj_name"])
    ex = _curvature(co, ctx["ea"], ctx["eb"]) - _rest_curv(obj, ctx)
    cw = np.clip((ex - REFINE_LO) / (REFINE_HI - REFINE_LO), 0.0, 1.0)
    cw = np.clip(_smooth(cw, ctx["ea"], ctx["eb"], 2) * 1.6, 0.0, 1.0)
    prot = np.isin(ctx["zone"], [C.ZID[k] for k in ("helper", "head", "hand", "foot")])
    cw[prot] = 0.0
    e = (_laplace(co, ctx["ea"], ctx["eb"], REFINE_ITERS) - co) * cw[:, None]
    m = np.linalg.norm(e, axis=1)
    e *= np.minimum(1.0, REFINE_CLAMP_M / np.maximum(m, 1e-9))[:, None]
    return e * min(amount, 1.0)


# v019 armpit_relax: the tuned v014 baseline carries 3 hard folds per armpit (front fold ~(+-186,-44,1262) mm, deep
# ~(+-182,-10,1263), back ~(+-197,38,1258); dihedral 82-125 deg).  A clamped Laplacian inside two soft ellipsoids per side,
# placed RELATIVE TO THE SHOULDER JOINT (so they follow every slider), removes all six without touching the measures
# (本机验证 on v018: creases 14 -> 8, 0 new issues, protected measures unchanged, max move 11 mm).
ARMPIT_ITERS = 20
ARMPIT_CLAMP_M = 0.012
ARMPIT_FRONT = ((0.004, 0.012, -0.058), (0.045, 0.070, 0.050))   # centre offset from the shoulder joint (|x|, y, z) / radii (m)
ARMPIT_BACK = ((0.018, 0.060, -0.061), (0.035, 0.035, 0.035))


def _armpit_w(co, ctx):
    s = co[ctx["shoulder_ids"]].mean(axis=0)
    ax = np.abs(co[:, 0])
    r = None
    for (ox, oy, oz), (rx, ry, rz) in (ARMPIT_FRONT, ARMPIT_BACK):
        q = np.stack([(ax - abs(s[0]) - ox) / rx, (co[:, 1] - s[1] - oy) / ry, (co[:, 2] - s[2] - oz) / rz], axis=1)
        n = np.linalg.norm(q, axis=1)
        r = n if r is None else np.minimum(r, n)
    t = np.clip(1.0 - r, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    w[ctx["zone"] == C.ZID["helper"]] = 0.0
    return w


def d_armpit_relax(co, ctx, amount):
    """POST: clamped Laplacian in the two armpit ellipsoids (front fold + hollow, back fold) of each side."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    w = _armpit_w(co, ctx)
    e = (_laplace(co, ctx["ea"], ctx["eb"], ARMPIT_ITERS) - co) * w[:, None]
    m = np.linalg.norm(e, axis=1)
    e *= np.minimum(1.0, ARMPIT_CLAMP_M / np.maximum(m, 1e-9))[:, None]
    return e * min(amount, 1.0)


# v020 back_relax: after armpit_relax the back still showed (user-reported 腋下/后背折痕) a diagonal fold running from the
# back armpit fold down the lateral scapula border (peak Laplacian excess 9.4 mm at (-17,+79,-48) mm from the shoulder joint,
# just OUTSIDE the armpit_relax back ellipsoid) and horizontal ripples along the trapezius top / upper scapula.  Same clamped
# Laplacian as armpit_relax, inside three soft capsules placed relative to the shoulder joint (|x| from the joint, y, z in mm).
# 本机验证 (scratchpad backfold_exp variant F on v019): 0 new health issues, the 8 protected measures unchanged, max move 12 mm.
BACK_RELAX_CAPS = (((-5, 75, -35), (-110, 80, -150), 0.040),     # back armpit fold -> down the lateral scapula border
                   ((-10, 20, 57), (-170, 75, 55), 0.040),       # trapezius top edge, out to the spine
                   ((-60, 85, 10), (-150, 85, 0), 0.045))        # upper scapula ripples


def _capsule_w(co, ctx, a, b, r):
    s = co[ctx["shoulder_ids"]].mean(axis=0)
    p = np.stack([np.abs(co[:, 0]) - abs(s[0]), co[:, 1] - s[1], co[:, 2] - s[2]], axis=1)
    a, b = np.asarray(a) / 1000.0, np.asarray(b) / 1000.0
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    u = np.clip(1.0 - np.linalg.norm(p - (a + t[:, None] * ab), axis=1) / r, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _back_relax_w(co, ctx):
    w = _armpit_w(co, ctx)                                   # the armpit ellipsoids too, so the two passes blend into one
    for a, b, r in BACK_RELAX_CAPS:
        w = np.maximum(w, _capsule_w(co, ctx, a, b, r))
    w[ctx["zone"] == C.ZID["helper"]] = 0.0
    return w


def d_back_relax(co, ctx, amount):
    """POST (after armpit_relax): clamped Laplacian in the back-fold / trapezius / upper-scapula capsules of each side."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    w = _back_relax_w(co, ctx)
    e = (_laplace(co, ctx["ea"], ctx["eb"], ARMPIT_ITERS) - co) * w[:, None]
    m = np.linalg.norm(e, axis=1)
    e *= np.minimum(1.0, ARMPIT_CLAMP_M / np.maximum(m, 1e-9))[:, None]
    return e * min(amount, 1.0)


# v019 arm_raise: a POSE, applied LAST (after every shape deformer and the point offsets), so all shape work is done in the
# A-pose and stays the same when the slider goes back to 0.  Each arm is rotated about its shoulder joint, in the plane of
# the upper arm and the horizontal (1 = upper arm horizontal = T-pose).  The rotation ANGLE (not the position) is blended:
# full on the arm, fading over the shoulder cap / armpit, 0 on the torso - angle blending keeps the shoulder from collapsing.
ARM_T0, ARM_T1 = -0.05, 0.10          # m along the upper-arm axis from the shoulder joint: weight 0 -> 1
ARM_BLEND_ITERS = 10
ARM_TORSO_R = 0.13                    # m: torso vertices farther than this from the shoulder joint never move


def _arm_side(co, ctx, s):
    obj = bpy.data.objects[ctx["obj_name"]]
    key = "arm_ids_" + s
    if key not in ctx:
        ctx[key] = (C.group_indices(obj, "joint-%s-shoulder" % s), C.group_indices(obj, "joint-%s-elbow" % s))
    js, je = ctx[key]
    return co[js].mean(axis=0), co[je].mean(axis=0)


def arm_raise_angle(co, ctx, s="l"):
    """angle (deg) between the upper arm and the horizontal - what arm_raise = 1 rotates"""
    J, E = _arm_side(co, ctx, s)
    a = (E - J) / np.linalg.norm(E - J)
    return math.degrees(math.asin(min(1.0, abs(a[2]))))


def d_arm_raise(co, ctx, amount):
    """POST, LAST: rotate both arms up about the shoulder joints (angle-blended weights)."""
    d = np.zeros_like(co)
    if amount <= 0:
        return d
    zone = ctx["zone"]
    arm = np.isin(zone, [C.ZID[k] for k in ("uarm", "larm", "hand")])
    torso = np.isin(zone, [C.ZID[k] for k in ("ribcage", "bust", "neck", "waist")])
    for s, sg in (("l", 1.0), ("r", -1.0)):
        J, E = _arm_side(co, ctx, s)
        a = (E - J) / np.linalg.norm(E - J)
        h = np.array([a[0], a[1], 0.0])
        h /= np.linalg.norm(h)
        n = np.cross(a, h)
        n /= np.linalg.norm(n)
        full = math.acos(float(np.clip(a @ h, -1.0, 1.0)))
        side = co[:, 0] * sg > 0.0
        t = (co - J) @ a
        u = np.clip((t - ARM_T0) / (ARM_T1 - ARM_T0), 0.0, 1.0)
        w = np.where(arm & side, u * u * (3.0 - 2.0 * u), 0.0)
        w = _smooth(w, ctx["ea"], ctx["eb"], ARM_BLEND_ITERS)
        w[arm & side & (u >= 1.0)] = 1.0                               # whole arm below the shoulder cap: rigid
        near = np.linalg.norm(co - J, axis=1) < ARM_TORSO_R
        w[~side | ~(arm | (torso & near))] = 0.0
        w = np.clip(w, 0.0, 1.0)
        th = full * min(float(amount), 1.0) * w                        # per-vertex angle
        v = co - J
        c, sn = np.cos(th)[:, None], np.sin(th)[:, None]
        rot = v * c + np.cross(n, v) * sn + n[None, :] * (v @ n)[:, None] * (1.0 - c)   # Rodrigues
        m = w > 0
        d[m] = (rot - v)[m]
    return d


# v019 RED points (红点 = bust / glute SHAPE design, the third tier after green = proportion and white = posture/local):
# anatomical PROFILE controls.  Each one is a smooth, signed (-1..1) displacement confined to one part of the bust mound or
# the buttock, built in a frame read from the CURRENT surface (nipple, inframammary fold, glute peak, gluteal fold).  They are
# designed to leave the green measurement lines alone (胸前凸 at 0.725 H, 胯宽 / 臀后凸 at 0.525 H, waist, legs) - the
# bands either stop before those lines or only move surface that does not define them - and they never touch the
# midline (sternum / cleft) in x.  All red deformers see the SAME input surface (they are summed, not chained), so the
# sliders are independent of each other's order.
RED_BUST = ("r_bust_upper", "r_bust_lower", "r_nip_pitch", "r_nip_yaw", "r_imf", "r_cleavage", "r_side_bust")
RED_GLUTE = ("r_glute_shelf", "r_glute_lowfull", "r_glute_fold_d", "r_glute_fold_h", "r_glute_hollow")
RED_IDS = RED_BUST + RED_GLUTE
# amplitudes (m at slider 1, before field smoothing).  Bounded by the neutral body's own near-threshold vertices (the hanging
# lower edge of the bust and the lateral inframammary fold sit at 7.9-9.6 mm Laplacian excess vs the 8 mm spike limit):
# bust x2 flagged new spikes there, so the bust stays at the first draft (r_imf + only clean at 0.009); the glute and the
# side-bust scanned clean at x2 alone, but with r_imf + it tipped the lateral fold crease (60.0-60.4 deg vs 60): 0.012.
R_UPPER_M, R_LOWER_M, R_PITCH_M, R_YAW_M = 0.015, 0.015, 0.012, 0.010
R_IMF_M, R_CLEAV_M, R_SIDE_M = 0.009, 0.010, 0.012
R_SHELF_M, R_LOWFULL_M, R_FOLD_D_M, R_FOLD_H_M, R_HOLLOW_M = 0.024, 0.024, 0.032, 0.024, 0.044
R_SHARP = 0.005             # m: a vertex farther than this from its neighbours' mean on the red input surface is "already sharp"
R_CREASE_DOT = -2.0         # ... or one of its faces is tilted > ~28 deg from the vertex normal (an existing crease)
R_GUARD_ITERS = 4           # harmonic passes of the red field on those vertices (see _red_guard)
R_NIP_R = 0.06              # m: radius of the nipple cap moved by r_nip_pitch / r_nip_yaw
R_FIELD_SMOOTH = 3          # Laplacian passes on each red displacement field (no kinks at the mask edges)
R_TOTAL_SMOOTH = 0          # extra passes on the SUM of the red fields - tried 4: it spreads the fields onto the fragile hanging
                            # lower edge of the bust and weakens every control by ~1/3, so it is OFF (kept for experiments)

# style presets (each sets ALL red sliders of its part; unlisted = 0)
BUST_PRESETS = {"水滴": dict(r_bust_upper=-0.6, r_bust_lower=0.7, r_nip_pitch=0.25, r_imf=-0.2, r_side_bust=0.2),
                "圆形": dict(r_bust_upper=0.7, r_bust_lower=0.5, r_cleavage=-0.3, r_side_bust=0.3),
                "上翘": dict(r_bust_upper=-0.3, r_bust_lower=0.4, r_nip_pitch=0.8, r_imf=0.2)}
GLUTE_PRESETS = {"蜜桃": dict(r_glute_shelf=0.6, r_glute_lowfull=0.6, r_glute_fold_d=0.4, r_glute_fold_h=0.3, r_glute_hollow=0.2),
                 "心形": dict(r_glute_shelf=-0.2, r_glute_lowfull=0.9, r_glute_fold_d=0.3, r_glute_hollow=-0.5),
                 "方形": dict(r_glute_lowfull=-0.3, r_glute_fold_d=-0.5, r_glute_fold_h=-0.2, r_glute_hollow=-0.9)}


def preset_state(part, name):
    """-> {red slider id: value} for a style preset ("原样" = all 0) of part "bust" / "glute"."""
    ids, table = (RED_BUST, BUST_PRESETS) if part == "bust" else (RED_GLUTE, GLUTE_PRESETS)
    p = table.get(name, {})
    return {k: float(p.get(k, 0.0)) for k in ids}


def _fan_tris(ctx):
    obj = bpy.data.objects[ctx["obj_name"]]
    me = obj.data
    ls = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), dtype=np.int32); me.loops.foreach_get("vertex_index", lv)
    out = []
    for k in range(1, int(lt.max()) - 1):
        m = lt > k + 1
        out.append(np.stack([lv[ls[m]], lv[ls[m] + k], lv[ls[m] + k + 1]], axis=1))
    return np.concatenate(out)


def _red_guard(base, dr, ctx):
    """On vertices that are ALREADY sharp on the input surface (the hanging lower edge of the bust, the lateral fold, the nipple
    tip...), replace the red displacement by the mean of its neighbours: a locally harmonic field adds ~nothing to the vertex's
    Laplacian, so the red controls move those spots along with their surroundings instead of sharpening them."""
    ea, eb = ctx["ea"], ctx["eb"]
    deg = ctx.get("_deg")
    if deg is None:
        deg = np.zeros(len(base)); _addat(deg, ea, 1); _addat(deg, eb, 1)
        deg = ctx["_deg"] = np.maximum(deg, 1.0)
    lap = np.linalg.norm(base - _nsum(base, ea, eb) / deg[:, None], axis=1)
    tri = ctx.get("_tri")
    if tri is None:
        tri = ctx["_tri"] = _fan_tris(ctx)
    fn = np.cross(base[tri[:, 1]] - base[tri[:, 0]], base[tri[:, 2]] - base[tri[:, 0]])
    vn = np.zeros_like(base)
    for j in range(3):
        _addat(vn, tri[:, j], fn)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    mind = np.ones(len(base))
    for j in range(3):
        np.minimum.at(mind, tri[:, j], (fn * vn[tri[:, j]]).sum(axis=1))
    sharp = np.flatnonzero((lap > R_SHARP) | (mind < R_CREASE_DOT))
    for _ in range(R_GUARD_ITERS):
        dr[sharp] = (_nsum(dr, ea, eb) / deg[:, None])[sharp]
    return dr


def _ss(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bump(t):
    """sin^2 bump: 0 with zero slope at t = 0 and t = 1, 1 at t = 0.5"""
    return np.sin(math.pi * np.clip(t, 0.0, 1.0)) ** 2


def _red_out(d, ctx):
    for k in range(3):
        if np.any(d[:, k]):
            d[:, k] = _smooth(d[:, k], ctx["ea"], ctx["eb"], R_FIELD_SMOOTH)
    return d


def _red_bust(co, ctx):
    """nipple frame + lateral offset from the nipple column (+ = toward the armpit), inframammary-fold depth L below the nipple"""
    c, dx, dz, wy = _bust_frame(co, ctx)
    lx = np.abs(co[:, 0]) - np.abs(c[:, 0])
    l, _ = _nipples(co, ctx)
    bz = (ctx["zone"] == C.ZID["bust"]) & (co[:, 0] > 0) & (np.abs(co[:, 0] - l[0]) < 0.02)
    L = max(float(l[2] - co[bz, 2].min()) if bz.any() else 0.07, 0.03)
    return c, lx, dz, wy, L


def _under(co, ctx, c, dz):
    """1 on the hidden underside of the breast (the inframammary fold region, behind the front 40 % of the nipple projection
    and below the nipple), 0 on the visible front.  Projection P = nipple depth in front of the sternum skin."""
    body = ctx["body"]
    B = co[body]
    cz, cy = float(c[:, 2].max()), float(c[:, 1].min())
    mid = (np.abs(B[:, 0]) < 0.012) & (np.abs(B[:, 2] - cz) < 0.03) & (B[:, 1] < 0)
    y_st = float(B[mid, 1].min()) if mid.any() else cy + 0.09
    P = max(y_st - cy, 0.03)
    behind = _ss((co[:, 1] - (cy + 0.4 * P)) / (0.3 * P))
    below = 1.0 - _ss((dz + 0.005) / 0.02)
    return behind * below


def d_r_bust_upper(co, ctx, amount):
    c, lx, dz, wy, L = _red_bust(co, ctx)
    U = max(float(co[ctx["shoulder_ids"], 2].mean() - c[:, 2].mean()), 0.05)
    v = _bump((dz / U - 0.2) / 0.75)
    px = 1.0 - _ss((np.abs(lx) - 0.03) / 0.05)
    d = np.zeros_like(co)
    d[:, 1] = -R_UPPER_M * amount * v * px * wy * ctx["torso_w"]
    return _red_out(d, ctx)


def d_r_bust_lower(co, ctx, amount):
    c, lx, dz, wy, L = _red_bust(co, ctx)
    v = _bump(-dz / L)
    px = 1.0 - _ss((np.abs(lx) - 0.03) / 0.05)
    d = np.zeros_like(co)
    d[:, 1] = -R_LOWER_M * amount * v * px * wy * ctx["torso_w"] * (1.0 - _under(co, ctx, c, dz))
    return _red_out(d, ctx)


def _nip_cap(co, ctx):
    c, lx, dz, wy, L = _red_bust(co, ctx)
    r = np.clip(np.sqrt(lx ** 2 + dz ** 2) / R_NIP_R, 0.0, 1.0)
    return (1.0 - r * r) ** 2 * wy * ctx["bust_w"] * (1.0 - _under(co, ctx, c, dz))


def d_r_nip_pitch(co, ctx, amount):
    d = np.zeros_like(co)
    d[:, 2] = R_PITCH_M * amount * _nip_cap(co, ctx)
    return _red_out(d, ctx)


def d_r_nip_yaw(co, ctx, amount):
    d = np.zeros_like(co)
    d[:, 0] = np.where(co[:, 0] >= 0, 1.0, -1.0) * R_YAW_M * amount * _nip_cap(co, ctx)
    return _red_out(d, ctx)


def d_r_imf(co, ctx, amount):
    c, lx, dz, wy, L = _red_bust(co, ctx)
    t = -dz / L
    w = _ss((t - 0.3) / 0.7) * (1.0 - _ss((t - 1.0) / 0.6))
    px = np.where(lx < 0, 1.0 - _ss((-lx - 0.015) / 0.03), 1.0 - _ss((lx - 0.05) / 0.05))   # medial side stops short of the under-bust spike verts
    d = np.zeros_like(co)
    d[:, 2] = R_IMF_M * amount * w * px * wy * ctx["torso_w"]
    return _red_out(d, ctx)


def d_r_cleavage(co, ctx, amount):
    c, lx, dz, wy, L = _red_bust(co, ctx)
    m = _ss((-lx - 0.02) / 0.04)
    mid = _ss((np.abs(co[:, 0]) - 0.004) / 0.03)
    vz = (1.0 - np.clip(np.abs(dz) / 0.08, 0.0, 1.0) ** 2) ** 2
    d = np.zeros_like(co)
    d[:, 0] = np.where(co[:, 0] >= 0, 1.0, -1.0) * R_CLEAV_M * amount * m * mid * vz * wy * ctx["torso_w"] * (1.0 - _under(co, ctx, c, dz))
    return _red_out(d, ctx)


def d_r_side_bust(co, ctx, amount):
    c, lx, dz, wy, L = _red_bust(co, ctx)
    lat = _ss((lx - 0.01) / 0.04) * (1.0 - _ss((lx - 0.08) / 0.04))
    vz = np.where(dz > 0, (1.0 - np.clip(dz / 0.05, 0.0, 1.0) ** 2) ** 2, (1.0 - np.clip(-dz / 0.05, 0.0, 1.0) ** 2) ** 2)
    w = R_SIDE_M * amount * lat * vz * wy * ctx["torso_w"] * (1.0 - _under(co, ctx, c, dz))
    d = np.zeros_like(co)
    d[:, 0] = np.where(co[:, 0] >= 0, 1.0, -1.0) * w
    d[:, 1] = -0.5 * w
    return _red_out(d, ctx)


def _red_glute(co, ctx):
    """height fraction fl, pelvis depth yc, back-facing weight, column weight around the glute peak, peak / fold / hip-joint heights (fl)"""
    body = ctx["body"]
    zmin = float(co[body, 2].min())
    H = float(co[body, 2].max()) - zmin
    fl = (co[:, 2] - zmin) / H
    P = co[ctx["hip_ids"]]
    yc = float(P[:, 1].mean())
    gz = (ctx["zone"] == C.ZID["glute"]) & (co[:, 0] > 0)
    G = co[gz]
    pk = G[np.argmax(G[:, 1])]
    col = gz & (np.abs(co[:, 0] - pk[0]) < 0.03)
    fl_p = (float(pk[2]) - zmin) / H
    fl_f = (float(co[col, 2].min()) - zmin) / H
    fl_h = (float(P[:, 2].mean()) - zmin) / H
    back = _ss((co[:, 1] - yc) / 0.05)
    px = 1.0 - _ss((np.abs(np.abs(co[:, 0]) - abs(float(pk[0]))) - 0.04) / 0.05)
    gw = np.clip(2.0 * ctx["glute_w2"], 0.0, 1.0)
    return fl, yc, back, px, gw, fl_p, fl_f, fl_h


def d_r_glute_shelf(co, ctx, amount):
    fl, yc, back, px, gw, fl_p, fl_f, fl_h = _red_glute(co, ctx)
    d = np.zeros_like(co)
    d[:, 1] = R_SHELF_M * amount * _bump((fl - fl_p) / 0.08) * back * px * gw
    return _red_out(d, ctx)


def d_r_glute_lowfull(co, ctx, amount):
    fl, yc, back, px, gw, fl_p, fl_f, fl_h = _red_glute(co, ctx)
    d = np.zeros_like(co)
    d[:, 1] = R_LOWFULL_M * amount * _bump((fl - fl_f + 0.005) / 0.08) * back * px * gw
    return _red_out(d, ctx)


def d_r_glute_fold_d(co, ctx, amount):
    fl, yc, back, px, gw, fl_p, fl_f, fl_h = _red_glute(co, ctx)
    d = np.zeros_like(co)
    d[:, 1] = -R_FOLD_D_M * amount * _bump((fl - fl_f + 0.02) / 0.04) * back * px * ctx["leg_w"]
    return _red_out(d, ctx)


def d_r_glute_fold_h(co, ctx, amount):
    fl, yc, back, px, gw, fl_p, fl_f, fl_h = _red_glute(co, ctx)
    d = np.zeros_like(co)
    d[:, 2] = R_FOLD_H_M * amount * _bump((fl - fl_f + 0.04) / 0.08) * back * px * ctx["leg_w"]
    return _red_out(d, ctx)


def d_r_glute_hollow(co, ctx, amount):
    fl, yc, back, px, gw, fl_p, fl_f, fl_h = _red_glute(co, ctx)
    ax = np.abs(co[:, 0])
    ry = co[:, 1] - yc
    post = _ss(ry / 0.03) * (1.0 - _ss((ry - 0.06) / 0.04))
    w = R_HOLLOW_M * amount * post * _ss((ax - 0.11) / 0.04) * _bump((fl - 0.54) / 0.09) * ctx["leg_w"]   # hip dip ABOVE the 0.525 H 胯宽 line, below the 0.65 H waist
    nx, ny = ax - 0.07, ry
    nn = np.sqrt(nx ** 2 + ny ** 2) + 1e-9
    d = np.zeros_like(co)
    d[:, 0] = -np.where(co[:, 0] >= 0, 1.0, -1.0) * w * nx / nn
    d[:, 1] = -w * ny / nn
    return _red_out(d, ctx)


_REF01_DELTA = None
_REF01_TOPOLOGY = None


def d_glute_anatomy_ref01(co, ctx, amount):
    """Reference-specific additive morph, after old glute smoothing; never replaces point offsets."""
    global _REF01_DELTA, _REF01_TOPOLOGY
    if amount <= 0:
        return np.zeros_like(co)
    if _REF01_DELTA is None:
        folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "custom_shapes"))
        manifest_path = os.path.join(folder, "glute_anatomy_ref01_v1.json")
        with open(manifest_path, encoding="utf-8") as f:
            meta = json.load(f)
        path = os.path.join(folder, "glute_anatomy_ref01_v1.npz")
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != meta["sha256"]:
            raise RuntimeError("参考01臀部形变资产校验失败，未执行修形")
        with np.load(path, allow_pickle=False) as archive:
            delta = archive["delta"].astype(np.float64)
        if not np.isfinite(delta).all():
            raise RuntimeError("参考01臀部形变包含非有限数值")
        _REF01_DELTA = delta
        _REF01_TOPOLOGY = meta["topology_sha256"]
    if _REF01_DELTA.shape != co.shape:
        raise RuntimeError("参考01臀部形变仅适用于原底模拓扑，当前顶点数量不匹配")
    if ctx["topology_sha256"] != _REF01_TOPOLOGY:
        raise RuntimeError("参考01臀部形变的面连接关系与当前底模不同，未执行修形")
    return _REF01_DELTA * min(float(amount), 1.0)


_DEFORMERS = {"legs_together": d_legs_together, "glute_proj": d_glute_proj, "knees_in": d_knees_in,
              "knees_back": d_knees_back, "glute_low": d_glute_low, "glute_round": d_glute_round,
              "bust_round": d_bust_round, "bust_upper": d_bust_upper, "bust_lower": d_bust_lower,
              "back_slim": d_back_slim, "back_anatomy": d_back_anatomy,
              "glute_anatomy_ref01": d_glute_anatomy_ref01,
              "bust_smooth": d_bust_smooth, "glute_smooth": d_glute_smooth,
              "nipple_flat": d_nipple_flat, "nipple_dome": d_nipple_dome,
              "bust_sag": d_bust_sag, "bust_yaw": d_bust_yaw, "bust_pitch": d_bust_pitch, "bust_lowfull": d_bust_lowfull,
              "glute_up": d_glute_up, "glute_leg": d_glute_leg, "armpit_relax": d_armpit_relax, "back_relax": d_back_relax, "refine_auto": d_refine_auto,
              "arm_raise": d_arm_raise,
              "r_bust_upper": d_r_bust_upper, "r_bust_lower": d_r_bust_lower, "r_nip_pitch": d_r_nip_pitch, "r_nip_yaw": d_r_nip_yaw,
              "r_imf": d_r_imf, "r_cleavage": d_r_cleavage, "r_side_bust": d_r_side_bust,
              "r_glute_shelf": d_r_glute_shelf, "r_glute_lowfull": d_r_glute_lowfull, "r_glute_fold_d": d_r_glute_fold_d,
              "r_glute_fold_h": d_r_glute_fold_h, "r_glute_hollow": d_r_glute_hollow}
# POST deformers run on the surface AFTER every additive deformer (MPFB + trial displacements).  They are smoothing
# passes and must see the finished additive surface.  nipple_flat runs LAST so it suppresses the nipple on the final,
# forward-projected mound (and after any bust_lower/smooth that could otherwise re-expose a tip).
_POST = ("bust_lower", "bust_smooth", "glute_smooth", "glute_anatomy_ref01", "nipple_flat", "nipple_dome", "glute_leg") + RED_IDS + (
    "armpit_relax", "back_relax", "refine_auto", "arm_raise")


def apply_custom(obj, full):
    """Recompute the cs_custom shape key from the current MPFB state (called at the end of s1_core.apply_state)."""
    vals = {k: float(full.get(k, 0.0)) for k in CUSTOM_IDS}
    keys = obj.data.shape_keys
    have_key = keys is not None and KEY in keys.key_blocks
    point_hook = getattr(C, "POINT_HOOK", None)         # stage-A control-point offsets (a_points.offsets_displacement)
    pd = point_hook(obj) if point_hook is not None else None
    ground = obj.get("cs_ground_z")                      # stage A: ground contact is a protected constraint (feet stay on the floor)
    if not have_key and pd is None and ground is None and not any(abs(v) > 1e-9 for v in vals.values()):
        return
    if keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
        keys = obj.data.shape_keys
    if KEY not in keys.key_blocks:
        obj.shape_key_add(name=KEY, from_mix=False)
    kb = keys.key_blocks[KEY]
    kb.value = 0.0
    kb.slider_min, kb.slider_max = 0.0, 1.0
    co0 = C.raw_coords(obj)                                # MPFB only
    ctx = _ctx(obj)
    d = np.zeros_like(co0)
    for k, fn in _DEFORMERS.items():
        if k not in _POST and abs(vals[k]) > 1e-9:
            d += fn(co0, ctx, vals[k])
    follow = getattr(C, "HELPER_FOLLOW", None)             # helper joint cubes follow the body around them (else the joint points drift off the legs)
    hmap = follow(obj) if follow is not None else None
    if hmap is not None:
        hz = C.load_zones(obj) == 0
        d[hz] = d[hmap[hz]]
    if pd is not None:
        d += pd
    red_base = None                                        # v019: every red deformer sees the same surface (summed, not chained)
    dr = None
    for k in _POST:
        if abs(vals[k]) > 1e-9:                            # v019: signed POST controls (red); the older POST controls are all >= 0
            if k in RED_IDS:
                if red_base is None:
                    red_base = co0 + d
                dp = _DEFORMERS[k](red_base, ctx, vals[k])
            else:
                dp = _DEFORMERS[k](co0 + d, ctx, vals[k])
            if k in RED_IDS:
                dr = dp if dr is None else dr + dp         # summed; smoothed once more as a whole below
            else:
                if hmap is not None:
                    dp[hz] = dp[hmap[hz]]
                d += dp
        if k == RED_IDS[-1] and dr is not None:
            for j in range(3 if R_TOTAL_SMOOTH else 0):
                if np.any(dr[:, j]):
                    dr[:, j] = _smooth(dr[:, j], ctx["ea"], ctx["eb"], R_TOTAL_SMOOTH)
            dr = _red_guard(red_base, dr, ctx)
            if hmap is not None:
                dr[hz] = 0.0                               # v019: red = surface shape only; the joint helpers (skeleton) stay
            d += dr
    if ground is not None:
        zmin = float((co0 + d)[ctx["body"], 2].min())
        d[:, 2] += float(ground) - zmin
    basis = np.empty(len(co0) * 3, dtype=np.float64)
    keys.reference_key.data.foreach_get("co", basis)
    kb.data.foreach_set("co", (basis.reshape(-1, 3) + d).astype(np.float32).reshape(-1))
    kb.value = 1.0
    obj.data.update()
    return d
