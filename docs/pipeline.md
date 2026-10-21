# Stages: what each does, entry scripts, outputs, gates

All paths are relative to the project root `<project root>`. Scripts in `scripts/` are shared modules. Scripts in `work/c/vNNN/` are one-off builders that produced version NNN. **Reuse them as templates; don't re-run old builders on new files blindly.** They often assert specific source versions.

Object names in the current character:
- `CS_Body`: MPFB basemesh, about 19k verts. It holds the shape keys, and the `cs_user_edit` key is the user's layer.
- `CS_FBHead`: the FaceBuilder head, joined in.
- `CS_Rig`: the Rigify rig, about 1090 bones.
- `CS_Jiggle`: the jiggle armature, at modifier index 0.
- `CS_Soft*`: the physics helpers.
- Vertex groups `CS_Part_{Bust|Glute}_{L|R}`.

---

## A. Body draft (v001–v016, user-confirmed)
- **Inputs** (`scripts/a_inputs.py`):
  - Reference images (front, side, back; tagged views).
  - A text prompt, parsed by rules into MPFB macro values.
  - Or a preset from `presets/`.
- **Build:**
  - `scripts/s1_core.py` holds the parameter model. Each state is a dict `{control: value}` stored as `cs_state` on the mesh, which is the single source of truth.
  - `scripts/s1_build.py` builds a new draft from a state JSON and never overwrites.
  - `scripts/s1_fit.py` fits MPFB controls to the reference silhouette numbers (`reference/ref01_metrics.json`).
- **Green points:** `scripts/a_points.py`, extended by the overlay `scripts/v3_points.py`.
  - About 28 body points: shoulder, waist, hip, knee, elbow, and the bust/glute mount points.
  - They drive the global shape with mirroring and protection constraints. Dragging the knee must not make the figure taller.
- **UI:** `scripts/s1_panel.py`, the N-panel tab "角色工作室". Launcher: `启动阶段1草模.bat`.
- **Gate:** overall height, weight, proportions and green-point behaviour.

## B. Body sculpt + parts (v017–v030)
- **White points:** `scripts/b_points.py`.
  - 101 points (52 on the left or midline, plus mirrors), with caps in `assets/white_point_caps.json`.
  - Each point has a semantic name, a falloff region and per-axis safe offset limits.
  - Dragging a point can back-solve a slider (for example, the outer deltoid point drives the shoulder-muscle slider).
- **Red points:** `scripts/s1_custom.py` (trial deformations) with the fixes in `scripts/v3_red.py`. There are 20 points driving 12 sliders for the bust and glute (upper, lower, inner, outer, orientation).
- **Separate parts:** `scripts/v3_parts.py`.
  - The bust and glute are independent meshes (pink and orange in the viewport), mounted at green points.
  - Later they are welded for real (stage E).
- **Interaction:** `scripts/v5_interaction.py`.
  - Keys: G move, X/Y/Z axis lock, typed mm values, Enter/Esc, Tab to cycle candidates, Shift for fine moves.
  - Undo and save-reopen must work.
- **Gate:** local shapes, part joins and point usability.

## C. Head and face
- **Route 1 (v031–v064): three-colour face points.**
  - `scripts/c_points.py`, `c_fit.py` (2D layout fit), `c_reference.py`, `c_ui.py` and `rw_*.py`.
  - About 110 semantic face points, documented in `docs/阶段C_完整点位.csv`.
  - Result: controllable, but **never reached likeness**, so it was abandoned for fitting.
- **Route 2 (v065+): FaceBuilder. This is the one to use.**
  1. The user fits the head in FaceBuilder in the GUI by pinning on 2–3 photos. The AI cannot pin in headless mode.
  2. Join the head to the body: `scripts/cc_v073_head.py` and `cc_v076_head.py`.
     - The head is a separate mesh that shares the body; the seam overlaps.
     - Feminising shape keys and a profile-driven chin come from `work/c/v073/build74.py` and `build75.py`.
  3. Neck seam: `scripts/cc_v078_fix.py` and `work/c/v076/v079_build.py`, which adds the trapezius lift and fills the nape.
- **Gate:** likeness from the front, side and three-quarter views, plus a neck-seam close-up.

## D. Skin, makeup, eyes (v060–v078)
- `scripts/cc_v068_skin.py`: the user's photo skin map on the head and neck.
- `scripts/cc_v069_skin.py`: whole-body skin, a cleaned face map, an editable makeup layer and brows.
- `work/c/v069/make_tex.py`: texture generation.
- `scripts/cc_v071_eyes.py`: eyeball scale relative to the lid opening.
- `scripts/cc_v078_fix.py`: matches the body base colour to the face photo. It must re-run after boot.
- Textures live in `textures/`.
- **Gate:** close-up look at the face, eyes and skin tone match.

## E. Topology and welding (v094–v100)
- **Bust/glute ring weld:** the dome rim is welded 1:1 to the body ring with a quad bridge. See `work/c/v095/weld95.py`, `work/c/v098/slide98.py` (ring position) and `work/c/v099/mid99b.py` (cleavage midline x = 0).
- **Texture fill under the welds:** `work/c/v094/texfill.py`.
- **Joint loops:** `work/c/v100/joints.py` adds concentric rings at the knee, ankle and elbow.
- **Body sliders from a phrase:** `work/c/v101/parse.py` with `muscle.py`, which turns a sentence like "肌肉型女性" into `cs_m_*`/`cs_s_*` keys.
- **Gate:** seams invisible in renders, clean bends.

## F. Rig and motion (v080–v089)
- **Rig:** `work/c/v080/v080_rig.py` runs MPFB `HumanService.add_builtin_rig(body,'rigify.human_toes')`, then `RigService.generate_rigify_rig`.
  - Head and helpers get weights by nearest-vertex transfer.
  - Teeth and tongue are parented to the head bone (`v084_anim.py`).
- **Mixamo:**
  - The user downloads the FBX files to `mocap/mixamo/`.
  - `work/c/v080/v088_mixamo.py` retargets onto Rigify FK using world-space rotation deltas, without an add-on. Arms are aimed because Mixamo rests in a T-pose and Rigify in an A-pose.
  - `v089_loop.py` cuts an in-place 29-frame catwalk loop with a crossfade.
- **Speed:** `v086_fast.py` turns the display subsurf off in the viewport and removes the material drivers (246 → 26 ms per frame).
- **Gate:** deformation in motion from the front and back.

## G. Physics (v102–v105, V105 on hold)
- **Jiggle bones:**
  - `work/c/v102/build.py` builds the `CS_Jiggle` armature at modifier 0.
  - `jig_bake.py` (also stored as the blend text `CS_Jiggle_Bake.py`) runs the spring simulation.
  - Parameters are the `jig_freq`, `jig_damp`, `jig_gain`, `jig_max` and `jig_rot` custom properties on the pose bones.
- **Soft body:** `work/c/v104/soft104.py` (blend text `CS_Soft_Build.py`).
  - A voxel cage per side (`CS_Soft_Bust_L/R`, `CS_Soft_Glute_L/R`) simulated as cloth with pressure.
  - Written back to the body with the geometry-nodes modifier `CS_SoftCopy`.
  - Inner-thigh contact uses the geometry-nodes modifier `CS_ThighContact`.
- **Hybrid (V105):** `work/c/v105cmp/tune.py` lowers the cage stiffness and re-enables the jiggle bones with a left/right frequency spread.
- **After changing the motion:** re-run `CS_Jiggle_Bake.py` and `CS_Soft_Build.py`.
- **Gate:** motion feel, left/right independence, no dents.

## H. Roadmap (not done; do in this order, one gate each)
1. **Static close-up:**
   - Pore normals, roughness zones and subsurface scattering.
   - Eyes with a cornea, iris depth and tear film.
   - Brows and lashes.
   - A Cycles close-up light rig.
2. **Expressions:** ARKit 52-style shape keys on `CS_FBHead`, plus blink, smile, visemes and eye gaze. Test that the face doesn't break.
3. **Hair:** Blender hair curves on the head, following the head bone, with light dynamics.
4. **Deformation:** corrective shape keys for the shoulder, hip and knee. Fix the arms going into the body in the back view of the Mixamo motion.
5. **Clothing:** one outfit with cloth simulation over the catwalk.
6. **Portfolio:** Cycles stills, a turntable, a short animation and a topology wireframe sheet.
7. **Optional:** re-attach white and red face points onto `CS_FBHead`, so the chain is photo fit, then point fine-tuning.
