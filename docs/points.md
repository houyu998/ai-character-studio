# Three-colour control-point spec

Sources: `docs/任务书_v3_结构人体与模块化零件.md`, `docs/任务书_v4_三色点与根部连接.md`, `docs/任务书_v6_第三部分_头面部构建.md`, `docs/任务书_v7_面部三色点与参考还原补充.md`.
Point tables:
- `docs/阶段B_参数归属表.md` (body white points).
- `docs/阶段C_完整点位.csv` and `docs/阶段C_v040_面部三色点.csv` (face).

## Roles

| Colour | Body | Face |
|---|---|---|
| Green: structure and mount points | Shoulder, waist, hip, knee and elbow positions. The bust/glute part mounts: the chest-centre green point is the engineering locator for both breasts, and the two breast roots sit apart on the chest front. Mount slots are reserved for future parts. | Head, both eyes, nose, mouth and both ears: overall layout and mounting (7 points) |
| White: surface | Shoulders and armpits, arms, back, belly and waist, legs, knees, calves, palms and feet. Fingers and toes are hidden by default. 101 points. | Skull, forehead, brow ridge, cheekbones, temples, jaw, chin, back of the head and ear roots. 18 main points + 57 advanced points. |
| Red: local detail | Bust upper, lower, inner and outer edges, plus orientation; glute upper, lower and outer edges. 20 points driving 12 sliders. | Eye corners, upper and lower lids, lid fold, eye bags and tear trough. Mouth corners, lip edges, lip peaks, philtrum and the lip seam. Nose tip, bridge, wings, columella and nostril edges. Ear helix, lobe and concha. |
| Purple | Problem markers (spikes, creases, intersections), not draggable | same |

## Behaviour requirements
- **Position:** each point sits on a stable anatomical feature, not on the "most responsive vertex". For example, the glute fold must not drift down the thigh.
- **Falloff:** smooth and limited, computed by topological distance and weights, not hand-drawn circles. No spikes, steps or intersections. Each axis has a measured safe range in mm.
- **Mirroring:** left and right are mirrored by default and the midline stays on x = 0. When mirroring is turned off and back on, ask the user: left→right, right→left, or average.
- **Sliders:** some points back-solve a slider. Sliders cover what is not intuitive as a point (fullness, muscularity, softness).
- **Editing:**
  - Mouse: select or drag.
  - Keys: G move, X/Y/Z lock an axis, type mm values (negative allowed), Enter to confirm, Esc to cancel, Shift for fine moves, Tab to cycle overlapping candidates.
  - Undo is required, and the state must survive save and reopen in a separate process.
- **Order of layers:** inherited body → shape presets / MPFB targets → green layout → white surface → red detail → detail layer → refit eyes, teeth, brows and hair. The user's `cs_user_edit` sits on top and is never erased.

## Lesson
Points edit **structure** well. They do not produce **likeness** from photos; use FaceBuilder or another photo fit for that, then points for fine adjustment.
