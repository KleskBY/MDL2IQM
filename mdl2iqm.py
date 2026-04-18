#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


IMG_EXTS = {".bmp", ".tga", ".jpg", ".jpeg", ".pcx", ".png"}


# ----------------------------
# Utilities
# ----------------------------

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed (exit {p.returncode}): {' '.join(cmd)}")
    return p


def clean_workdir(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    for item in workdir.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try:
                item.unlink()
            except Exception:
                pass


def find_first_qc(workdir: Path) -> Path:
    qcs = sorted(workdir.rglob("*.qc"))
    if not qcs:
        raise FileNotFoundError(f"No .qc found in: {workdir}")
    return qcs[0]


def resolve_file_recursive(workdir: Path, filename: str) -> Path:
    direct = workdir / filename
    if direct.exists():
        return direct
    matches = sorted(workdir.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing file referenced by QC: {filename}")


# ----------------------------
# QC parsing (multi-mesh + sequences)
# ----------------------------

@dataclass
class QcSequence:
    qc_name: str     # first name in $sequence "idle" ...
    smd_base: str    # second token, used as smd stem
    fps: int
    loop: int        # 1 if loop present else 0


def parse_qc(qc_path: Path) -> Tuple[List[str], List[QcSequence]]:
    """
    Supports:
      $body "Name" "meshstem"
      $bodygroup "x" { studio "meshstem" ... }
      $sequence "name" "animstem" fps 30 [loop]
    Returns:
      mesh_smds: ["mesh1.smd", "mesh2.smd", ...] (unique, in order)
      sequences: list of QcSequence in QC order (not deduped)
    """
    mesh_candidates: List[str] = []
    sequences: List[QcSequence] = []

    body_re = re.compile(r'^\s*\$body\s+"[^"]+"\s+"([^"]+)"', re.IGNORECASE)
    bodygroup_start_re = re.compile(r'^\s*\$bodygroup\b', re.IGNORECASE)
    bodygroup_studio_re = re.compile(r'^\s*studio\s+"([^"]+)"', re.IGNORECASE)

    seq_re = re.compile(r'^\s*\$sequence\s+"([^"]+)"\s+"([^"]+)"(.*)$', re.IGNORECASE)
    fps_re = re.compile(r'\bfps\s+([0-9]+)\b', re.IGNORECASE)
    loop_re = re.compile(r'\bloop\b', re.IGNORECASE)

    lines = qc_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_bodygroup = False

    for raw in lines:
        if bodygroup_start_re.match(raw):
            in_bodygroup = True

        if in_bodygroup:
            m = bodygroup_studio_re.match(raw)
            if m:
                mesh_candidates.append(m.group(1) + ".smd")
            if "}" in raw:
                in_bodygroup = False

        m = body_re.match(raw)
        if m:
            mesh_candidates.append(m.group(1) + ".smd")
            continue

        m = seq_re.match(raw)
        if m:
            qc_name = m.group(1)
            smd_base = m.group(2)
            tail = m.group(3) or ""

            mfps = fps_re.search(tail)
            fps = int(mfps.group(1)) if mfps else 30
            loop = 1 if loop_re.search(tail) else 0

            sequences.append(QcSequence(qc_name=qc_name, smd_base=smd_base, fps=fps, loop=loop))

    # dedupe mesh list, preserve order
    seen = set()
    mesh_smds: List[str] = []
    for m in mesh_candidates:
        k = m.lower()
        if k not in seen:
            seen.add(k)
            mesh_smds.append(m)

    return mesh_smds, sequences


# ----------------------------
# SMD: rewrite materials
# ----------------------------

def smd_is_material_line(line: str) -> bool:
    s = line.strip()
    if not s or s.lower() == "end":
        return False
    # material lines are typically single tokens (no spaces)
    if any(c.isspace() for c in s):
        return False
    # vertex lines start with int + spaces
    if re.match(r"^-?\d+\s", s):
        return False
    return True


def strip_root_xy(smd_path: Path) -> None:
    """
    Modifies animation SMD in-place: root bone (bone 0) X and Y are locked
    to their frame-0 values for every frame, removing horizontal locomotion.
    Z and all rotations are preserved.
    """
    bone_re = re.compile(r'^(\s*0\s+)(-?[\d.eE+]+)(\s+)(-?[\d.eE+]+)(\s+.*)$')

    ref_x: Optional[float] = None
    ref_y: Optional[float] = None
    in_skeleton = False
    out: List[str] = []

    for raw in smd_path.read_text(encoding="utf-8", errors="ignore").splitlines(True):
        stripped = raw.strip().lower()
        if stripped == "skeleton":
            in_skeleton = True
            out.append(raw)
            continue
        if stripped == "end" and in_skeleton:
            in_skeleton = False
            out.append(raw)
            continue
        if in_skeleton:
            m = bone_re.match(raw)
            if m:
                x, y = float(m.group(2)), float(m.group(4))
                if ref_x is None:
                    ref_x, ref_y = x, y  # lock to frame-0 position
                raw = f"{m.group(1)}{ref_x:.6f}{m.group(3)}{ref_y:.6f}{m.group(5)}"
                if not raw.endswith('\n'):
                    raw += '\n'
        out.append(raw)

    smd_path.write_text("".join(out), encoding="utf-8", errors="ignore")
    print(f"   [inplace] stripped root XY from: {smd_path.name}")


def rewrite_smd_materials(smd_path: Path, modelname: str, prefix: str = "textures/models") -> Tuple[int, Set[str]]:
    """
    Rewrites material lines inside triangles section to:
      textures/models/<modelname>/<stem>.png

    Returns:
      (replacements_count, texture_stems_used)
    """
    text = smd_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    out: List[str] = []
    in_triangles = False
    replaced = 0
    stems_used: Set[str] = set()

    for raw in text:
        if not in_triangles:
            out.append(raw)
            if raw.strip().lower() == "triangles":
                in_triangles = True
            continue

        stripped = raw.strip()
        if stripped.lower() == "end":
            in_triangles = False
            out.append(raw)
            continue

        if smd_is_material_line(raw):
            mat = stripped.replace("\\", "/")
            base = mat.split("/")[-1]
            stem = Path(base).stem
            stems_used.add(stem.lower())

            new_mat = f"{prefix}/{modelname}/{stem}.png"
            out.append(new_mat + ("\n" if raw.endswith("\n") else ""))
            replaced += 1
        else:
            out.append(raw)

    if replaced:
        smd_path.write_text("".join(out), encoding="utf-8", errors="ignore")

    return replaced, stems_used


# ----------------------------
# SMD: count frames for animations
# ----------------------------

def count_smd_frames(smd_path: Path) -> int:
    time_re = re.compile(r'^\s*time\s+(-?\d+)\s*$', re.IGNORECASE)
    count = 0
    for line in smd_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if time_re.match(line):
            count += 1
    if count == 0:
        raise RuntimeError(f"No 'time N' frames found in SMD: {smd_path}")
    return count


# ----------------------------
# framegroups writer
# ----------------------------

def write_framegroups(out_framegroups: Path, entries: List[Tuple[int, int, int, int]]) -> None:
    """
    line format:
      start_frame frame_count fps loop
    """
    out_framegroups.write_text(
        "".join(f"{start} {count} {fps} {loop}\n" for (start, count, fps, loop) in entries),
        encoding="utf-8"
    )


# ----------------------------
# Image conversion + texture staging
# ----------------------------

def convert_images_to_png(workdir: Path, magick_exe: Path, delete_originals: bool, chunk: int = 200) -> None:
    imgs = [p for p in workdir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        print("== No textures found to convert ==")
        return

    print(f"== Converting {len(imgs)} image(s) to PNG (mogrify) ==")

    # Avoid Windows command line length issues by chunking
    for i in range(0, len(imgs), chunk):
        batch = imgs[i:i + chunk]
        cmd = [str(magick_exe), "mogrify", "-format", "png"] + [str(p) for p in batch]
        run_cmd(cmd)

    if delete_originals:
        for p in imgs:
            if p.suffix.lower() != ".png":
                try:
                    p.unlink()
                except Exception:
                    pass


def stage_textures_as_png(
    workdir: Path,
    content_root: Path,
    modelname: str,
    stems_needed: Set[str],
) -> int:
    """
    Copies converted PNGs from workdir into:
      <content_root>/textures/models/<modelname>/<stem>.png

    It matches by stem (case-insensitive). Returns number of textures copied.
    """
    if not stems_needed:
        return 0

    dest_dir = content_root / "textures" / "models" / modelname
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Build stem -> source png mapping
    pngs = [p for p in workdir.rglob("*.png") if p.is_file()]
    src_map: Dict[str, Path] = {}
    for p in pngs:
        stem = p.stem.lower()
        # keep first occurrence; you can change priority if needed
        if stem not in src_map:
            src_map[stem] = p

    copied = 0
    missing: List[str] = []

    for stem in sorted(stems_needed):
        src = src_map.get(stem.lower())
        if not src:
            missing.append(stem)
            continue
        dst = dest_dir / f"{stem}.png"
        shutil.copy2(src, dst)
        copied += 1

    if missing:
        print("!! Missing texture PNGs for stems:")
        for m in missing[:30]:
            print("   -", m)
        if len(missing) > 30:
            print("   ...")

    print(f"== Textures staged to: {dest_dir} ({copied} copied) ==")
    return copied


# ----------------------------
# Optional: merge multiple mesh SMDs into one
# ----------------------------

def split_smd_sections(smd_path: Path) -> Tuple[List[str], List[str]]:
    """
    Returns:
      header_lines: everything up to and including 'triangles\\n'
      tri_lines: lines between 'triangles' and final 'end' (excluding that 'end')
    """
    lines = smd_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    header: List[str] = []
    tri: List[str] = []
    in_tri = False

    for raw in lines:
        if not in_tri:
            header.append(raw)
            if raw.strip().lower() == "triangles":
                in_tri = True
            continue

        # in triangles
        if raw.strip().lower() == "end":
            in_tri = False
            # stop reading after first triangles block end (mesh SMDs should only have one)
            break
        tri.append(raw)

    if not any(l.strip().lower() == "triangles" for l in header):
        raise RuntimeError(f"SMD has no triangles section: {smd_path}")

    return header, tri


def merge_mesh_smds(mesh_paths: List[Path], out_path: Path, strict: bool = True) -> None:
    """
    Merges multiple mesh SMDs by:
      - taking header (nodes+skeleton+triangles line) from first
      - appending triangles blocks from all meshes
      - ending with 'end\\n'

    Assumption (usually true for HL viewmodels/weapons): nodes+skeleton are identical.
    If strict=True and headers differ (excluding whitespace), it raises.
    """
    if len(mesh_paths) < 2:
        shutil.copy2(mesh_paths[0], out_path)
        return

    h0, t0 = split_smd_sections(mesh_paths[0])

    def normalize_header(h: List[str]) -> str:
        # normalize whitespace for comparison
        return "".join([re.sub(r"\s+", " ", x.strip()) + "\n" for x in h])

    h0_norm = normalize_header(h0)

    merged_tri: List[str] = []
    for mp in mesh_paths:
        h, t = split_smd_sections(mp)
        if strict:
            if normalize_header(h) != h0_norm:
                raise RuntimeError(
                    "Mesh SMD headers differ; cannot safely merge.\n"
                    f"First: {mesh_paths[0].name}\n"
                    f"Diff:  {mp.name}\n"
                    "Use --force-merge-meshes to merge anyway (may break bones)."
                )
        merged_tri.extend(t)

    out_path.write_text("".join(h0) + "".join(merged_tri) + "end\n", encoding="utf-8", errors="ignore")


# ----------------------------
# Main
# ----------------------------
def move_outputs_to_content_root(
    iqm_path: Path,
    content_root: Path,
    modelname: str
) -> Tuple[Path, Path]:
    """
    Moves:
      <model>.iqm
      <model>.iqm.framegroups
    into:
      <content_root>/models/

    Returns:
      (new_iqm_path, new_framegroups_path)
    """
    models_dir = content_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    new_iqm = models_dir / iqm_path.name
    new_iqm_fg = models_dir / (iqm_path.name + ".framegroups")

    shutil.move(str(iqm_path), new_iqm)

    fg = iqm_path.with_suffix(iqm_path.suffix + ".framegroups")
    if fg.exists():
        shutil.move(str(fg), new_iqm_fg)

    return new_iqm, new_iqm_fg

def main() -> int:
    ap = argparse.ArgumentParser(description="Half-Life MDL -> IQM pipeline (mdldec + ImageMagick + iqm) with QC-driven sequences.")
    ap.add_argument("mdl", help="Path to input .mdl")

    ap.add_argument("--mdldec", default="mdldec.exe", help="Path to mdldec.exe (default: mdldec.exe)")
    ap.add_argument("--magick", default="magick.exe", help="Path to magick.exe (default: magick.exe)")
    ap.add_argument("--iqm", default="iqm.exe", help="Path to iqm tool (default: iqm.exe)")

    ap.add_argument("--workdir", default="./work", help="Working directory (default: ./work)")
    ap.add_argument("--outdir", default=".", help="Output directory for IQM + framegroups (default: .)")
    ap.add_argument("--out", default=None, help="Explicit output IQM path (default: <outdir>/<mdlname>.iqm)")

    ap.add_argument("--rotate", default="0,0,90", help="iqm rotation '-r x,y,z' (default: 0,0,90)")
    ap.add_argument("--keep-work", action="store_true", help="Do not wipe workdir before running")
    ap.add_argument("--delete-original-textures", action="store_true", help="After PNG conversion, delete non-PNG extracted textures")

    ap.add_argument("--qc", default=None, help="Explicit QC path (otherwise first .qc in workdir)")
    ap.add_argument("--material-prefix", default="textures/models", help="Material prefix in SMD (default: textures/models)")

    # Texture staging
    ap.add_argument("--content-root", default=None, help="If set, copy textures to <content-root>/textures/models/<modelname>/")
    ap.add_argument("--no-copy-textures", action="store_true", help="Disable copying textures even if --content-root is set")

    # In-place animation (strip root XY locomotion)
    ap.add_argument("--no-inplace", action="store_true",
                    help="Disable in-place root-XY stripping (default: stripping is ON)")

    # Multi-mesh support
    ap.add_argument("--merge-meshes-on-fail", action="store_true",
                    help="If iqm fails with multiple meshes, merge mesh SMDs and retry.")
    ap.add_argument("--force-merge-meshes", action="store_true",
                    help="Merge meshes even if SMD headers differ (risky). Only used with --merge-meshes-on-fail.")

    args = ap.parse_args()

    mdl_path = Path(args.mdl).resolve()
    if not mdl_path.exists():
        print(f"ERROR: MDL not found: {mdl_path}", file=sys.stderr)
        return 2

    workdir = Path(args.workdir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out).resolve() if args.out else (outdir / f"{mdl_path.stem}.iqm")

    mdldec_exe = Path(args.mdldec)
    magick_exe = Path(args.magick)
    iqm_exe = Path(args.iqm)

    modelname = mdl_path.stem

    print(f"== MDL:      {mdl_path}")
    print(f"== Workdir:  {workdir}")
    print(f"== Out IQM:  {out_path}")

    # 1) Clean workdir
    if not args.keep_work:
        print("== Cleaning workdir ==")
        clean_workdir(workdir)
    else:
        workdir.mkdir(parents=True, exist_ok=True)

    # 2) Run mdldec
    print("== Running mdldec ==")
    run_cmd([str(mdldec_exe), str(mdl_path), str(workdir)])

    # 3) Convert images to PNG
    convert_images_to_png(workdir, magick_exe, delete_originals=args.delete_original_textures)

    # 4) Parse QC for meshes + sequences
    qc_path = Path(args.qc).resolve() if args.qc else find_first_qc(workdir)
    print(f"== Using QC: {qc_path}")

    mesh_names, sequences = parse_qc(qc_path)
    if not mesh_names:
        raise RuntimeError("No mesh SMDs found in QC ($body/$bodygroup).")
    if not sequences:
        print("!! Warning: No $sequence entries found in QC (model may be static).")

    mesh_paths = [resolve_file_recursive(workdir, m) for m in mesh_names]
    anim_paths = [resolve_file_recursive(workdir, s.smd_base + ".smd") for s in sequences]

    print(f"== Meshes: {len(mesh_paths)}")
    for p in mesh_paths:
        print("   -", p.name)
    print(f"== Sequences: {len(sequences)}")

    # 5) Rewrite materials in ALL mesh SMDs
    print("== Rewriting SMD material names ==")
    stems_needed: Set[str] = set()
    total_replaced = 0

    for mp in mesh_paths:
        replaced, stems = rewrite_smd_materials(mp, modelname, prefix=args.material_prefix)
        total_replaced += replaced
        stems_needed |= stems

    # Anim SMDs usually don't have triangles; but safe to run (will be 0 most of the time)
    for apath in set(anim_paths):  # de-dupe identical paths
        replaced, stems = rewrite_smd_materials(apath, modelname, prefix=args.material_prefix)
        total_replaced += replaced
        stems_needed |= stems

    print(f"== Material lines rewritten: {total_replaced} ==")

    # 5b) Strip root XY (in-place animations) — on by default, disable with --no-inplace
    if not args.no_inplace:
        print("== Making animations in-place (strip root XY) ==")
        for anim_path in set(anim_paths):
            strip_root_xy(anim_path)

    # 6) Copy textures into content root (optional)
    if args.content_root and not args.no_copy_textures:
        content_root = Path(args.content_root).resolve()
        print(f"== Staging textures to content root: {content_root} ==")
        stage_textures_as_png(workdir, content_root, modelname, stems_needed)

    # 7) Generate framegroups (based on QC order + anim SMD frame counts)
    if sequences:
        print("== Generating framegroups ==")
        start = 0
        entries: List[Tuple[int, int, int, int]] = []

        for seq, anim_path in zip(sequences, anim_paths):
            frames = count_smd_frames(anim_path)
            entries.append((start, frames, seq.fps, seq.loop))
            start += frames

        framegroups_path = out_path.with_suffix(out_path.suffix + ".framegroups")  # <name>.iqm.framegroups
        write_framegroups(framegroups_path, entries)
        print(f"== Wrote: {framegroups_path} ==")
    else:
        print("== Skipping framegroups (no sequences) ==")

    # 8) Run iqm tool: multi-mesh, with fallback merge if requested
    print("== Building IQM ==")
    cmd = [str(iqm_exe), "-r", args.rotate, str(out_path)]
    cmd += [str(p) for p in mesh_paths]
    cmd += [str(p) for p in anim_paths]
    p = run_cmd(cmd, check=False)

    if len(mesh_paths) > 1:
        print("== Attempting mesh merge and retry ==")
        merged_mesh = workdir / f"{modelname}__merged_mesh.smd"
        merge_mesh_smds(mesh_paths, merged_mesh, strict=not args.force_merge_meshes)

        cmd2 = [str(iqm_exe), "-r", args.rotate, str(out_path), str(merged_mesh)]
        cmd2 += [str(p) for p in anim_paths]
        run_cmd(cmd2, check=True)


    # 9) Move final outputs into content root (optional)
    if args.content_root:
        content_root = Path(args.content_root).resolve()
        print(f"== Moving outputs to content root: {content_root} ==")

        new_iqm, new_fg = move_outputs_to_content_root(out_path, content_root, modelname)

        print(f"== IQM moved to: {new_iqm} ==")
        if new_fg.exists():
            print(f"== Framegroups moved to: {new_fg} ==")

    print("DONE.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        raise SystemExit(1)
