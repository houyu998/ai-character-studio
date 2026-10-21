# Traps learned the hard way

## Files and process
- Users save hand edits as `*_1.blend` or as autosaves. Before building, check for newer user files and build on them.
- When Blender crashes, autosaves are in `%TEMP%` as `<name>_<pid>_autosave.blend`.
- Never `taskkill` all blender.exe processes; the user may have an unsaved session open.
- C: has little free space; each version is about 32 MB. Archive old versions, don't delete them without asking.

## Rendering
- A relative output path inside Blender resolves against C:/. Always pass absolute paths.
- Fixed lights produce false defects, such as "open back of head" or a pale arc on a dress (grazing specular). Cross-check with a second light setup before "fixing" geometry.
- AgX desaturates red lips; compensate in the material, not the geometry.

## Head, face and skin
- A boot script can re-hook the `CS_C_Skin` base chain (V060–V068 layers). Guard it with the `cc_v078` flag and re-apply `cc_v078_fix`.
- Over-narrowing the jaw makes a chin shadow (v053 was rejected).
- ControlNet or depth-estimation output is not a ruler: the median error was 2.6 mm.
- FaceBuilder pinning is a manual GUI task for the user.

## Rig and motion
- Blender 4.4: when you change `animation_data.action`, also set `action_slot` (`a.slots[0]`).
- Mixamo rests in a T-pose and Rigify in an A-pose, so aim the arms; don't copy local rotations directly.
- The Mixamo catwalk walks out and back, so start = end does not mean in-place. Cut one cycle and remove the root travel.
- Display subsurf plus material drivers made the viewport take 246 ms per frame. Turn both off in the viewport only.

## Welding
- Weld bust/glute domes to body rings 1:1 (same vertex count). Bridges with mismatched counts crease.
- Texture fill under welded areas: erosion can eat into valid skin, so check the boundary.

## Physics
- The Armature modifier normalises weights. Add a root bone weight of 1 − Σ, or tiny jiggle weights jump to full.
- Fine-mesh cloth on the bust collapsed the top and creased. Use a coarse voxel cage (about 3 cm) with internal pressure, then copy the motion to the body with geometry nodes.
- Two-way cage collision between the left and right sides creates a dependency cycle. Use one-way collision (the right side collides with the left).
- Cage damping and mass barely change the amplitude; stiffness (bend, compression, pressure) does. Bust mass above about 1.0 brings creases back.
- Cloth rebakes are not reproducible exactly, so compare variants within one batch.
- Unhidden physics helpers (`CS_Soft_Collider` and the cages) look black and blotchy in the viewport. They are harmless and must not be deleted.
