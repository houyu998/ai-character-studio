# mocap/

Motion clips are NOT included. Mixamo's terms do not allow redistributing the downloaded files.

1. Sign in at https://www.mixamo.com with your own Adobe account.
2. Pick a clip (e.g. Idle, Catwalk). Download as FBX, "Without Skin", 30 fps.
3. Save it here as `mocap/mixamo/<ClipName>.fbx`.
4. Run the retarget step described in `docs/pipeline.md` (stage F).

The `.gitignore` keeps any `.fbx` in this folder out of the repository.
