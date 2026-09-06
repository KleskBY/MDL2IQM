# MDL2IQM
A tool to convert Half-life MDL models to IQM

## Scripts

| script | what |
| --- | --- |
| `mdl2iqm.py` | the converter: decompiles an MDL, bakes bodygroups, strips root motion on flagged sequences, writes the IQM plus its `.framegroups` / `.bodyinfo` / `.skin` sidecars |
| `hl_retarget.py` | rebuilds an HL model so it keeps its own mesh and skeleton but plays another model's animations. World-space transfer, so rigs that disagree on bone names (`R Arm1` vs `R UpperArm`, `L Leg` vs `L Thigh`) still reproduce the pose exactly |
| `iqm_renameanim.py` | renames a sequence in an already-built IQM, patching the anim table, `.framegroups` and `.bodyinfo` together. In-place string patch, so the new name cannot be longer than the old one; prefer `mdl2iqm.py --rename-anim` when reconverting is an option |
| `mdlinfo.py` | MDL binary reader (sequence table, motion flags, skin families) used by the converter |
