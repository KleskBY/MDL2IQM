#!/usr/bin/env python3
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import mdlinfo


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


def sanitize_token(n: str) -> str:
    """One whitespace-free token (framegroups/skin/bodyinfo files split on whitespace)."""
    return re.sub(r"\s+", "_", n.strip()) if n else ""


# ----------------------------
# QC parsing (bodygroups + sequences)
# ----------------------------

@dataclass
class QcSequence:
    qc_name: str     # first name in $sequence "idle" ...
    smd_base: str    # second token, used as smd stem (first blend anim)
    fps: int
    loop: int        # 1 if loop present else 0
    has_motion: bool = False  # LX/LY/LZ token on the $sequence line
    blend_smds: List[str] = field(default_factory=list)  # ALL anim stems (>=1)
    blend_start: float = 0.0  # blend range from `blend XR <start> <end>`
    blend_end: float = 0.0


@dataclass
class QcBodyGroup:
    name: str
    studios: List[str] = field(default_factory=list)  # smd stems, in QC order


def parse_qc(qc_path: Path) -> Tuple[List[QcBodyGroup], List[QcSequence]]:
    """
    Supports:
      $body "Name" "meshstem"                     -> its own group of one
      $bodygroup "x" { studio "a" studio "b" ... blank }
      $sequence "name" "animstem" fps 30 [loop]
    Returns:
      groups: bodygroups in QC order, each with ALL studio stems
      sequences: list of QcSequence in QC order (not deduped)
    """
    groups: List[QcBodyGroup] = []
    sequences: List[QcSequence] = []

    body_re = re.compile(r'^\s*\$body\s+"([^"]+)"\s+"([^"]+)"', re.IGNORECASE)
    bodygroup_start_re = re.compile(r'^\s*\$bodygroup\s+"?([^"\s{]+)"?', re.IGNORECASE)
    bodygroup_studio_re = re.compile(r'^\s*studio\s+"([^"]+)"', re.IGNORECASE)
    bodygroup_blank_re = re.compile(r'^\s*blank\b', re.IGNORECASE)

    # name, then ALL consecutive quoted tokens = blend anim smd stems
    # (mdldec: $sequence "aim" "aim_blend1" "aim_blend2" blend XR -45 45 fps 14 loop)
    seq_re = re.compile(r'^\s*\$sequence\s+"([^"]+)"((?:\s+"[^"]+")+)(.*)$', re.IGNORECASE)
    fps_re = re.compile(r'\bfps\s+([0-9]+)\b', re.IGNORECASE)
    loop_re = re.compile(r'\bloop\b', re.IGNORECASE)
    blend_re = re.compile(r'\bblend\s+\w+\s+(-?[0-9.]+)\s+(-?[0-9.]+)', re.IGNORECASE)
    # case-sensitive: LX/LY/LZ motion-extraction tokens (mdldec re-emits them)
    motion_re = re.compile(r'\bL[XYZ]\b')

    lines = qc_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cur_group: Optional[QcBodyGroup] = None

    for raw in lines:
        m = bodygroup_start_re.match(raw)
        if m:
            cur_group = QcBodyGroup(name=sanitize_token(m.group(1)))
            groups.append(cur_group)
            # fall through: '{' and studios may share the line

        if cur_group is not None:
            sm = bodygroup_studio_re.match(raw)
            if sm:
                cur_group.studios.append(sm.group(1))
            elif bodygroup_blank_re.match(raw):
                # HL 'blank' submodel: empty geometry slot. Track with a
                # sentinel so body-value indexing stays aligned.
                cur_group.studios.append("")
            if "}" in raw:
                cur_group = None
            continue

        m = body_re.match(raw)
        if m:
            groups.append(QcBodyGroup(name=sanitize_token(m.group(1)), studios=[m.group(2)]))
            continue

        m = seq_re.match(raw)
        if m:
            qc_name = m.group(1)
            smd_stems = re.findall(r'"([^"]+)"', m.group(2))
            tail = m.group(3) or ""

            mfps = fps_re.search(tail)
            fps = int(mfps.group(1)) if mfps else 30
            loop = 1 if loop_re.search(tail) else 0
            mblend = blend_re.search(tail)
            bstart = float(mblend.group(1)) if mblend else 0.0
            bend = float(mblend.group(2)) if mblend else 0.0

            sequences.append(QcSequence(qc_name=qc_name, smd_base=smd_stems[0], fps=fps,
                                        loop=loop, has_motion=bool(motion_re.search(tail)),
                                        blend_smds=smd_stems,
                                        blend_start=bstart, blend_end=bend))

    return groups, sequences


# ----------------------------
# SMD: rewrite materials
# ----------------------------

def smd_is_material_line(line: str) -> bool:
    s = line.strip()
    if not s or s.lower() == "end":
        return False
    # vertex lines start with int + spaces; everything else in a triangles
    # block is a material. HL texture names can contain spaces ("Clone of
    # bootside.bmp") so multi-token lines are still materials.
    if re.match(r"^-?\d+\s", s):
        return False
    return True


def parse_smd_nodes(smd_path: Path) -> List[Tuple[int, str, int]]:
    """Reads the `nodes` block: (id, name, parent) in file order."""
    node_re = re.compile(r'^\s*(-?\d+)\s+"([^"]*)"\s+(-?\d+)\s*$')
    nodes: List[Tuple[int, str, int]] = []
    in_nodes = False
    for raw in smd_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw.strip().lower()
        if stripped == "nodes":
            in_nodes = True
            continue
        if in_nodes:
            if stripped == "end":
                break
            m = node_re.match(raw)
            if m:
                nodes.append((int(m.group(1)), m.group(2), int(m.group(3))))
    return nodes


def _is_lower_body_bone(name: str) -> bool:
    n = name.lower()
    return (n == "bip01" or "pelvis" in n
            or "leg" in n or "foot" in n or "toe" in n)


def detect_split_bone(nodes: List[Tuple[int, str, int]]) -> Optional[Tuple[int, str]]:
    """
    Finds the first torso bone (legs/torso boundary) for the QC-side
    upper/lower body split. Emitted as `splitbone <id> <name>` in .bodyinfo:
    the id is the 0-based SMD/IQM joint index — QC owns any engine ±1.

    Requires a legs-first skeleton (root, pelvis, leg chains, then torso).
    Returns None (with a loud warning) when the ordering doesn't hold —
    a reorder pass is future work; today's HL biped rigs are all legs-first.
    """
    if not nodes:
        return None

    by_name = {name.lower(): (bid, name) for bid, name, _ in nodes}
    split = by_name.get("bip01 spine")
    if split is None:
        # fallback: first non-leg child of the pelvis
        pelvis_ids = {bid for bid, name, _ in nodes if "pelvis" in name.lower()}
        for bid, name, parent in nodes:
            if parent in pelvis_ids and "leg" not in name.lower():
                split = (bid, name)
                break
    if split is None:
        print("!! splitbone: no 'Bip01 Spine' or non-leg pelvis child found; not emitted")
        return None

    sid, sname = split
    bad_low = [name for bid, name, _ in nodes if bid < sid and not _is_lower_body_bone(name)]
    bad_high = [name for bid, name, _ in nodes
                if bid >= sid and ("leg" in name.lower() or "foot" in name.lower()
                                   or "toe" in name.lower())]
    if bad_low or bad_high:
        print("!! splitbone: skeleton is not legs-first; splitbone not emitted "
              "(reorder pass = future work)")
        for n in bad_low[:8]:
            print(f"   - non-leg bone below split: {n}")
        for n in bad_high[:8]:
            print(f"   - leg bone above split: {n}")
        return None

    print(f"== splitbone: {sid} \"{sname}\" (legs = bones 0..{sid - 1}) ==")
    return sid, sname


def offset_root_bone(smd_path: Path, dx: float, dy: float, dz: float) -> None:
    """
    Modifies SMD in-place: bone 0 (root) X/Y/Z are shifted by (dx, dy, dz)
    in every `time N` block. SMD bone positions are parent-relative, so
    shifting the root translates the entire skeleton — and the mesh verts
    weighted to it — by the offset. Use this instead of `iqm.exe -t`,
    which only translates mesh verts and leaves the skeleton at origin
    (mesh detaches from bones).

    Works on both mesh SMDs (single `time 0` reference pose) and anim SMDs
    (per-frame `time N`).
    """
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        return

    bone_re = re.compile(r'^(\s*0\s+)(-?[\d.eE+]+)(\s+)(-?[\d.eE+]+)(\s+)(-?[\d.eE+]+)(\s+.*)$')

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
                x = float(m.group(2)) + dx
                y = float(m.group(4)) + dy
                z = float(m.group(6)) + dz
                raw = f"{m.group(1)}{x:.6f}{m.group(3)}{y:.6f}{m.group(5)}{z:.6f}{m.group(7)}"
                if not raw.endswith('\n'):
                    raw += '\n'
        out.append(raw)

    smd_path.write_text("".join(out), encoding="utf-8", errors="ignore")
    print(f"   [offset] root bone shifted by ({dx},{dy},{dz}) in: {smd_path.name}")


def strip_root_xy(smd_path: Path) -> None:
    """
    LEGACY (--inplace): root bone (bone 0) X and Y locked to frame-0 values
    for every frame. Blanket removal of horizontal locomotion — breaks
    animations that legitimately move the root. Prefer the per-sequence
    extract_linear_motion() driven by MDL motiontype flags.
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


def extract_linear_motion(smd_path: Path, axes: Set[int], eps: float = 0.05) -> Tuple[float, float, float]:
    """
    Per-sequence root-motion extraction for sequences flagged STUDIO_LX/LY/LZ:
    measure the root bone's total drift from first to last frame and subtract
    a linear ramp (frame/(numframes-1)) * drift. In-frame sway is preserved;
    net locomotion is removed.

    The motiontype flags gate WHICH sequences get processed; the drifting
    axes are measured, not taken from the flags — SMD root-bone positions
    are in the bone's local space, which is rotated relative to the world
    axes the LX/LY/LZ flags describe (e.g. HL bipeds store world-X motion
    as local -Y).

    If the drift is ~0 (already in-place) this is a no-op. Returns the
    measured drift (x, y, z).
    """
    bone_re = re.compile(r'^(\s*0\s+)(-?[\d.eE+]+)(\s+)(-?[\d.eE+]+)(\s+)(-?[\d.eE+]+)(\s+.*)$')
    time_re = re.compile(r'^\s*time\s+(-?\d+)\s*$', re.IGNORECASE)

    lines = smd_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    # Pass 1: collect root-bone positions per frame (skeleton section only)
    in_skeleton = False
    frame_idx = -1
    root_pos: List[Tuple[float, float, float]] = []
    for raw in lines:
        stripped = raw.strip().lower()
        if stripped == "skeleton":
            in_skeleton = True
            frame_idx = -1
            continue
        if stripped == "end" and in_skeleton:
            in_skeleton = False
            continue
        if not in_skeleton:
            continue
        if time_re.match(raw):
            frame_idx += 1
            continue
        m = bone_re.match(raw)
        if m and frame_idx >= 0:
            while len(root_pos) <= frame_idx:
                root_pos.append(None)
            if root_pos[frame_idx] is None:  # bone 0 appears once per frame
                root_pos[frame_idx] = (float(m.group(2)), float(m.group(4)), float(m.group(6)))

    n = len(root_pos)
    if n < 2 or root_pos[0] is None or root_pos[-1] is None:
        return (0.0, 0.0, 0.0)

    drift = tuple(root_pos[-1][i] - root_pos[0][i] for i in range(3))
    active = [i for i in range(3) if abs(drift[i]) > eps]
    if not active:
        return drift

    # Pass 2: subtract the ramp on active axes
    out: List[str] = []
    in_skeleton = False
    frame_idx = -1
    for raw in lines:
        stripped = raw.strip().lower()
        if stripped == "skeleton":
            in_skeleton = True
            frame_idx = -1
            out.append(raw)
            continue
        if stripped == "end" and in_skeleton:
            in_skeleton = False
            out.append(raw)
            continue
        if in_skeleton:
            if time_re.match(raw):
                frame_idx += 1
                out.append(raw)
                continue
            m = bone_re.match(raw)
            if m and frame_idx >= 0:
                t = frame_idx / (n - 1)
                pos = [float(m.group(2)), float(m.group(4)), float(m.group(6))]
                for i in active:
                    pos[i] -= drift[i] * t
                raw = (f"{m.group(1)}{pos[0]:.6f}{m.group(3)}{pos[1]:.6f}"
                       f"{m.group(5)}{pos[2]:.6f}{m.group(7)}")
                if not raw.endswith('\n'):
                    raw += '\n'
        out.append(raw)

    smd_path.write_text("".join(out), encoding="utf-8", errors="ignore")
    axis_names = "".join("xyz"[i] for i in active)
    print(f"   [motion] extracted linear {axis_names} drift "
          f"({drift[0]:.1f},{drift[1]:.1f},{drift[2]:.1f}) from: {smd_path.name}")
    return drift


# ----------------------------
# Blend-sequence flattening (HL pev->blending emulation)
# ----------------------------
# HL blended sequences ($sequence ... "a_blend1" "a_blend2" blend XR -45 45)
# store multiple anims; the game picks the mix at runtime from aim pitch.
# IQM has one anim per sequence, and QC can't blend — so bake the anim at
# blend VALUE 0 (degrees), exactly what HLMV does on sequence change
# (SetBlending(0, 0.f) -> setting 127 -> ~50/50 slerp). Without this the
# first blend anim alone is used = blendstart = full-down aim (~45deg tilt).

def _euler_to_quat(rx: float, ry: float, rz: float) -> Tuple[float, float, float, float]:
    """SMD bone rotation (radians, R = Rz*Ry*Rx) -> quaternion (w,x,y,z)."""
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    return (cz * cy * cx + sz * sy * sx,
            cz * cy * sx - sz * sy * cx,
            cz * sy * cx + sz * cy * sx,
            sz * cy * cx - cz * sy * sx)


def _quat_to_euler(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    w, x, y, z = q
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    return (math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            math.asin(sinp),
            math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_slerp(q0, q1, t: float):
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = tuple(-v for v in q1)
        dot = -dot
    if dot > 0.9995:  # near-parallel: nlerp
        q = tuple(a + t * (b - a) for a, b in zip(q0, q1))
        n = math.sqrt(sum(v * v for v in q)) or 1.0
        return tuple(v / n for v in q)
    th0 = math.acos(min(1.0, dot))
    s0 = math.sin((1.0 - t) * th0) / math.sin(th0)
    s1 = math.sin(t * th0) / math.sin(th0)
    return tuple(s0 * a + s1 * b for a, b in zip(q0, q1))


def _parse_smd_skeleton(path: Path) -> Tuple[List[str], List[List[Tuple[int, List[float]]]]]:
    """
    Returns (header_lines up to+incl 'skeleton', frames), where each frame is
    a list of (bone_index, [px py pz rx ry rz]) in file order.
    """
    header: List[str] = []
    frames: List[List[Tuple[int, List[float]]]] = []
    in_skeleton = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw.strip()
        if not in_skeleton:
            header.append(raw)
            if stripped.lower() == "skeleton":
                in_skeleton = True
            continue
        if stripped.lower() == "end":
            break
        if stripped.lower().startswith("time"):
            frames.append([])
            continue
        parts = stripped.split()
        if len(parts) >= 7 and frames:
            frames[-1].append((int(parts[0]), [float(v) for v in parts[1:7]]))
    return header, frames


def blend_smd_pair(path_a: Path, path_b: Path, frac: float, out_path: Path) -> None:
    """Writes an anim SMD whose skeleton is a per-bone lerp(pos)/slerp(rot)
    of the two inputs at `frac` (0 = all A, 1 = all B)."""
    header, frames_a = _parse_smd_skeleton(path_a)
    _, frames_b = _parse_smd_skeleton(path_b)

    n = min(len(frames_a), len(frames_b))
    if len(frames_a) != len(frames_b):
        print(f"!! blend anims differ in frame count ({len(frames_a)} vs "
              f"{len(frames_b)}), using first {n}: {path_a.name} / {path_b.name}")

    out: List[str] = list(header)
    for fi in range(n):
        out.append(f"time {fi}")
        fb = dict(frames_b[fi])
        for bone, va in frames_a[fi]:
            vb = fb.get(bone, va)
            px = va[0] + frac * (vb[0] - va[0])
            py = va[1] + frac * (vb[1] - va[1])
            pz = va[2] + frac * (vb[2] - va[2])
            q = _quat_slerp(_euler_to_quat(va[3], va[4], va[5]),
                            _euler_to_quat(vb[3], vb[4], vb[5]), frac)
            rx, ry, rz = _quat_to_euler(q)
            out.append(f"{bone}  {px:.6f} {py:.6f} {pz:.6f}  {rx:.6f} {ry:.6f} {rz:.6f}")
    out.append("end")
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def flatten_blend_sequence(seq: QcSequence, smd_paths: List[Path], workdir: Path) -> Path:
    """
    Bakes a multi-blend sequence into one anim SMD at blend value 0 degrees.
    t = position of 0 within [blend_start, blend_end]; anims span that range
    evenly, so f = t*(N-1) picks the bracketing pair + slerp fraction.
    (9-blend 3x3 aim grids: t=0.5 -> f=4 -> exact center anim, no slerp.)
    """
    n = len(smd_paths)
    span = seq.blend_end - seq.blend_start
    t = 0.5 if abs(span) < 0.001 else (0.0 - seq.blend_start) / span
    t = max(0.0, min(1.0, t))

    f = t * (n - 1)
    i = min(int(math.floor(f + 1e-6)), n - 1)
    frac = f - i

    out_path = workdir / f"{sanitize_token(seq.qc_name)}__blendmid.smd"
    if frac < 1e-4 or i >= n - 1:
        shutil.copy2(smd_paths[i], out_path)
        print(f"   [blend] {seq.qc_name}: {n} blend anims, picked #{i + 1} "
              f"(blend {seq.blend_start:g}..{seq.blend_end:g}, value 0)")
    else:
        blend_smd_pair(smd_paths[i], smd_paths[i + 1], frac, out_path)
        print(f"   [blend] {seq.qc_name}: {n} blend anims, slerp #{i + 1}/#{i + 2} "
              f"@ {frac:.2f} (blend {seq.blend_start:g}..{seq.blend_end:g}, value 0)")
    return out_path


def sanitize_mesh_token(s: str) -> str:
    """Lowercase [a-z0-9_] token — safe for engine surface-name buffers
    (64-char qpath limits) and .skin file parsing (no parens/commas/slashes)."""
    return re.sub(r"[^a-z0-9_]", "", s.lower())


def rewrite_smd_materials(
    smd_path: Path,
    modelname: str,
    prefix: str = "textures/models",
    mesh_tag: str = "",
) -> Tuple[int, Set[str], List[Tuple[str, str]]]:
    """
    Rewrites material lines inside the triangles section.

    Normal mode (no mesh_tag): material becomes a direct texture path,
      textures/models/<modelname>/<stem>.png
    which the engine resolves without shaders.

    Bodygroup mode (mesh_tag set): material becomes a SHORT unique token,
      <tag>_<stem-sanitized>          e.g. "b1s2_sci3back2"
    The iqm tool groups triangles by material and uses it as the mesh name,
    so this keeps every submodel a separate, individually hideable surface
    even when submodels share textures. Kept short and free of parens/paths
    because engines cap surface names (64-char qpath) and .skin parsing
    chokes on exotic characters — the generated .skin files then map each
    token to its real texture path (or common/nodraw). Without a skin file
    applied these meshes have no resolvable texture, but bodygroup models
    always ship skin files.

    Returns:
      (replacements_count, texture_stems_used, mesh_list)
      mesh_list: unique (iqm_mesh_name, texture_stem) pairs for this SMD
    """
    text = smd_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    out: List[str] = []
    in_triangles = False
    replaced = 0
    stems_used: Set[str] = set()
    mesh_list: List[Tuple[str, str]] = []
    mesh_seen: Set[str] = set()

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

            if mesh_tag:
                # short token; no extension so IQM mesh name == material
                new_mat = f"{mesh_tag}_{sanitize_mesh_token(stem)}"
                mesh_name = new_mat
            else:
                # spaces would split the material into multiple SMD tokens —
                # staging writes the PNG under the same underscored name
                safe_stem = re.sub(r"\s+", "_", stem)
                mesh_name = f"{prefix}/{modelname}/{safe_stem}"
                new_mat = mesh_name + ".png"
            if mesh_name not in mesh_seen:
                mesh_seen.add(mesh_name)
                mesh_list.append((mesh_name, stem.lower()))

            out.append(new_mat + ("\n" if raw.endswith("\n") else ""))
            replaced += 1
        else:
            out.append(raw)

    if replaced:
        smd_path.write_text("".join(out), encoding="utf-8", errors="ignore")

    return replaced, stems_used, mesh_list


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

def write_framegroups(out_framegroups: Path, entries: List[Tuple[int, int, int, int, str]]) -> None:
    """
    line format:
      start_frame frame_count fps loop name

    `name` is whitespace-sanitized (spaces → underscore) so the 5th column
    is one token — FTE's framegroups parser splits on whitespace.
    """
    out_framegroups.write_text(
        "".join(
            f"{start} {count} {fps} {loop} {sanitize_token(name)}\n"
            for (start, count, fps, loop, name) in entries
        ),
        encoding="utf-8"
    )


# ----------------------------
# events writer (HL mstudioevent_t -> QC-readable sidecar)
# ----------------------------

def write_events(out_events: Path, entries: List[Tuple[int, int, int, int, str]],
                 info: "mdlinfo.MdlInfo") -> bool:
    """
    Emit <model>.iqm.events: one line per anim event, QC tokenize-friendly:

      groupname time_seconds event_code options

    groupname matches the .framegroups 5th column (sanitized). time is the
    event's frame divided by the sequence fps. options is '-' when empty
    (QC needs a fixed token count). Returns False (and writes nothing) when
    no sequence carries events.
    """
    lines: List[str] = []
    for (_start, _count, fps, _loop, name) in entries:
        seq = info.sequence_by_label(name)
        if not seq or not seq.events:
            continue
        use_fps = fps if fps > 0 else (seq.fps if seq.fps > 0 else 15)
        for ev in seq.events:
            t = ev.frame / use_fps
            opts = sanitize_token(ev.options) if ev.options else "-"
            lines.append(f"{sanitize_token(name)} {t:.3f} {ev.event} {opts}\n")
    if not lines:
        return False
    out_events.write_text("".join(lines), encoding="utf-8")
    return True


# ----------------------------
# Bodygroups: skin files + bodyinfo sidecar
# ----------------------------

@dataclass
class SubmodelMeshes:
    group_index: int     # bodygroup index
    submodel_index: int  # index within the group
    stem: str            # smd stem ("" = HL 'blank' slot, no meshes)
    meshes: List[Tuple[str, str]] = field(default_factory=list)  # (iqm_mesh_name, tex_stem)


def compute_group_bases(groups: List[QcBodyGroup], info: mdlinfo.MdlInfo) -> List[int]:
    """
    HL body encoding: submodel = (body / base) % nummodels per group.
    Use the authoritative bases from the MDL binary when group structure
    matches the QC parse; otherwise recompute in QC order and warn.
    """
    if len(info.bodyparts) == len(groups) and all(
        len(bp.models) == len(g.studios) for bp, g in zip(info.bodyparts, groups)
    ):
        return [bp.base for bp in info.bodyparts]

    print("!! QC bodygroup structure differs from MDL binary; recomputing bases from QC order")
    bases = []
    base = 1
    for g in groups:
        bases.append(base)
        base *= max(1, len(g.studios))
    return bases


def write_skin_files(
    out_iqm: Path,
    groups: List[QcBodyGroup],
    bases: List[int],
    submodels: List[SubmodelMeshes],
    info: mdlinfo.MdlInfo,
    modelname: str,
    prefix: str,
    mesh_filter: Optional[Set[str]] = None,
) -> List[Path]:
    """
    Emits <out_iqm.name>_<i>.skin next to out_iqm for every body×family
    combination, index = body * numfamilies + family (mirrored by QC
    Body_Apply). Every IQM mesh is listed (DP renders unlisted meshes as
    checkerboard): visible submodels point at their (family-remapped)
    texture, hidden submodels at common/nodraw.

    mesh_filter: when set (--split-body halves), only meshes named in the
    set are listed — each half's skin file addresses its own IQM.
    """
    total_combos = 1
    for g in groups:
        total_combos *= max(1, len(g.studios))
    numfam = info.numskinfamilies

    written: List[Path] = []
    for body in range(total_combos):
        chosen = {}  # group_index -> submodel_index
        for gi, g in enumerate(groups):
            count = max(1, len(g.studios))
            chosen[gi] = (body // bases[gi]) % count

        for fam in range(numfam):
            remap = info.family_texture_map(fam)
            lines: List[str] = []
            for sm in submodels:
                visible = chosen[sm.group_index] == sm.submodel_index
                for mesh_name, tex_stem in sm.meshes:
                    if mesh_filter is not None and mesh_name not in mesh_filter:
                        continue
                    if visible:
                        # sanitized: HL texture names carry parens etc. that
                        # break engine .skin parsing / shader lookup — the
                        # staged PNGs use the same sanitized names
                        stem = sanitize_mesh_token(remap.get(tex_stem, tex_stem))
                        lines.append(f"{mesh_name},{prefix}/{modelname}/{stem}")
                    else:
                        lines.append(f"{mesh_name},common/nodraw")

            idx = body * numfam + fam
            skin_path = out_iqm.with_name(f"{out_iqm.name}_{idx}.skin")
            skin_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written.append(skin_path)

    print(f"== Wrote {len(written)} skin file(s) ({total_combos} bodies x {numfam} families) ==")
    return written


def write_bodyinfo(
    out_iqm: Path,
    groups: List[QcBodyGroup],
    bases: List[int],
    info: mdlinfo.MdlInfo,
    sequences: List[QcSequence],
    splitbone: Optional[Tuple[int, str]] = None,
) -> Path:
    """
    Sidecar consumed by QC (BodyInfo_Load in serverqc/util_body.qc):
      families <numfamilies>
      splitbone <id> <name>          (0-based joint id of first TORSO bone)
      group <name> <count> <base>
      anim <name> <lm_x> <lm_y> <lm_z>
    """
    lines = [f"families {info.numskinfamilies}"]
    if splitbone is not None:
        lines.append(f"splitbone {splitbone[0]} {sanitize_token(splitbone[1])}")
    for g, base in zip(groups, bases):
        lines.append(f"group {sanitize_token(g.name)} {max(1, len(g.studios))} {base}")
    for seq in sequences:
        s = info.sequence_by_label(seq.qc_name)
        if s and any(abs(v) > 0.01 for v in s.linearmovement):
            lm = s.linearmovement
            lines.append(f"anim {sanitize_token(seq.qc_name)} {lm[0]:.2f} {lm[1]:.2f} {lm[2]:.2f}")

    path = out_iqm.with_name(out_iqm.name + ".bodyinfo")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"== Wrote: {path.name} ==")
    return path


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
    tagged_copies: Optional[List[Tuple[str, str]]] = None,
) -> int:
    """
    Copies converted PNGs from workdir into:
      <content_root>/textures/models/<modelname>/<stem>.png

    tagged_copies: extra (dest_stem, source_stem) pairs — bodygroup-mode
    baked-in material names (<tag>__<stem>) get a real texture so combo 0
    renders even without skin-file support.

    Matches by stem (case-insensitive). Returns number of textures copied.
    """
    if not stems_needed and not tagged_copies:
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

    for dest_stem, source_stem in (tagged_copies or []):
        src = src_map.get(source_stem.lower())
        if not src:
            missing.append(source_stem)
            continue
        dst = dest_dir / f"{dest_stem}.png"
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
# IQM feet-height measurement (--feet-z)
# ----------------------------
# Minimal IQM v2 reader: skins every vertex at a given animation frame and
# returns the global minimum Z. Used to place the model's animated standing
# feet exactly at the hull floor (Quake player hull: -24) instead of trusting
# a hand-tuned --offset constant.

def _quat_to_mat3(x: float, y: float, z: float, w: float) -> List[List[float]]:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _mat34_compose(t, q, s) -> List[List[float]]:
    r = _quat_to_mat3(*q)
    return [[r[i][0] * s[0], r[i][1] * s[1], r[i][2] * s[2], t[i]] for i in range(3)]


def _mat34_mul(a, b) -> List[List[float]]:
    return [
        [
            a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
            + (a[i][3] if j == 3 else 0.0)
            for j in range(4)
        ]
        for i in range(3)
    ]


def _mat34_invert(m) -> List[List[float]]:
    a, b, c = m[0][0], m[0][1], m[0][2]
    d, e, f = m[1][0], m[1][1], m[1][2]
    g, h, i = m[2][0], m[2][1], m[2][2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        det = 1e-12
    inv3 = [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]
    tx, ty, tz = m[0][3], m[1][3], m[2][3]
    return [
        inv3[r][:3] + [-(inv3[r][0] * tx + inv3[r][1] * ty + inv3[r][2] * tz)]
        for r in range(3)
    ]


def measure_animated_feet_z(iqm_path: Path, frame_index: int = 0) -> float:
    """Skins all vertices at global animation frame `frame_index` (post
    `-r` rotation, i.e. final engine space) and returns the minimum Z."""
    import struct

    data = iqm_path.read_bytes()
    if data[:16] != b"INTERQUAKEMODEL\x00":
        raise RuntimeError(f"Not an IQM file: {iqm_path}")
    hdr = struct.unpack_from("<27I", data, 16)
    (_version, _filesize, _flags, _num_text, _ofs_text, _num_meshes, _ofs_meshes,
     num_vertexarrays, num_vertexes, ofs_vertexarrays, _num_triangles, _ofs_triangles,
     _ofs_adjacency, num_joints, ofs_joints, num_poses, ofs_poses, _num_anims,
     _ofs_anims, num_frames, num_framechannels, ofs_frames, _ofs_bounds,
     _num_comment, _ofs_comment, _num_ext, _ofs_ext) = hdr

    if num_joints == 0:
        raise RuntimeError("IQM has no joints")

    # joints: bind pose
    joints = []
    for j in range(num_joints):
        vals = struct.unpack_from("<Ii10f", data, ofs_joints + j * 48)
        parent = vals[1]
        t = vals[2:5]
        q = vals[5:9]   # x y z w
        s = vals[9:12]
        joints.append((parent, t, q, s))

    bind_world: List[List[List[float]]] = []
    for parent, t, q, s in joints:
        local = _mat34_compose(t, q, s)
        bind_world.append(_mat34_mul(bind_world[parent], local) if parent >= 0 else local)
    inv_bind = [_mat34_invert(m) for m in bind_world]

    # poses + frame data -> animated world matrices at frame_index
    if num_poses and num_frames:
        frame_index = max(0, min(frame_index, num_frames - 1))
        poses = []
        for p in range(num_poses):
            vals = struct.unpack_from("<iI20f", data, ofs_poses + p * 88)
            poses.append((vals[0], vals[1], vals[2:12], vals[12:22]))
        fofs = ofs_frames + frame_index * num_framechannels * 2
        fdata = struct.unpack_from(f"<{num_framechannels}H", data, fofs)
        k = 0
        anim_world: List[List[List[float]]] = []
        for parent, mask, coffset, cscale in poses:
            ch = []
            for c in range(10):
                v = coffset[c]
                if mask & (1 << c):
                    v += cscale[c] * fdata[k]
                    k += 1
                ch.append(v)
            local = _mat34_compose(ch[0:3], ch[3:7], ch[7:10])
            anim_world.append(_mat34_mul(anim_world[parent], local) if parent >= 0 else local)
    else:
        anim_world = bind_world

    skin_mats = [_mat34_mul(anim_world[j], inv_bind[j]) for j in range(num_joints)]

    # vertex arrays
    IQM_POSITION, IQM_BLENDINDEXES, IQM_BLENDWEIGHTS = 0, 4, 5
    pos_arr = idx_arr = wgt_arr = None
    for a in range(num_vertexarrays):
        vtype, _vflags, vformat, vsize, vofs = struct.unpack_from(
            "<5I", data, ofs_vertexarrays + a * 20)
        if vtype == IQM_POSITION:
            pos_arr = (vformat, vsize, vofs)
        elif vtype == IQM_BLENDINDEXES:
            idx_arr = (vformat, vsize, vofs)
        elif vtype == IQM_BLENDWEIGHTS:
            wgt_arr = (vformat, vsize, vofs)
    if pos_arr is None:
        raise RuntimeError("IQM has no position vertex array")

    minz = None
    pofs = pos_arr[2]
    for v in range(num_vertexes):
        px, py, pz = struct.unpack_from("<3f", data, pofs + v * 12)
        if idx_arr is not None and wgt_arr is not None:
            bidx = struct.unpack_from("<4B", data, idx_arr[2] + v * 4)
            bwgt = struct.unpack_from("<4B", data, wgt_arr[2] + v * 4)
            z = 0.0
            total = 0.0
            for bi, bw in zip(bidx, bwgt):
                if bw == 0:
                    continue
                w = bw / 255.0
                m = skin_mats[bi]
                z += w * (m[2][0] * px + m[2][1] * py + m[2][2] * pz + m[2][3])
                total += w
            if total <= 0.0:
                m = skin_mats[bidx[0]]
                z = m[2][0] * px + m[2][1] * py + m[2][2] * pz + m[2][3]
        else:
            m = skin_mats[0]
            z = m[2][0] * px + m[2][1] * py + m[2][2] * pz + m[2][3]
        if minz is None or z < minz:
            minz = z

    if minz is None:
        raise RuntimeError("IQM has no vertices")
    return minz


# ----------------------------
# Body-half mesh splitting (--split-body)
# ----------------------------
# Emits <name>_legs.iqm / <name>_torso.iqm for the DP/Zircon two-entity
# player split: full skeleton + full anim set in both (identical framegroup
# indices), meshes partitioned per-TRIANGLE by dominant vertex bone weight.
# Triangles straddling the boundary go into BOTH halves — a co-planar
# overlap ring that hides the waist seam.

def smd_vertex_primary_bone(vert_line: str) -> int:
    """Dominant-weight bone of an SMD triangle vertex line:
    <bone> px py pz nx ny nz u v [numlinks (<bone> <weight>)*]
    Unlisted remainder weight belongs to the leading <bone>."""
    toks = vert_line.split()
    base = int(toks[0])
    if len(toks) < 10:
        return base
    try:
        nlinks = int(toks[9])
    except ValueError:
        return base
    pairs = []
    total = 0.0
    for i in range(nlinks):
        try:
            b = int(toks[10 + i * 2])
            w = float(toks[11 + i * 2])
        except (IndexError, ValueError):
            break
        pairs.append((b, w))
        total += w
    if total < 1.0 - 1e-4:
        pairs.append((base, 1.0 - total))
    if not pairs:
        return base
    return max(pairs, key=lambda p: p[1])[0]


def split_mesh_smd(merged_mesh: Path, splitbone: int,
                   legs_out: Path, torso_out: Path) -> Tuple[Set[str], Set[str]]:
    """Partitions a (material-rewritten) mesh SMD at `splitbone` (0-based;
    legs = bones < splitbone). Header (nodes + skeleton + 'triangles') is
    copied verbatim into both halves. Returns (legs_meshes, torso_meshes):
    the material/mesh names present in each half."""
    header, tri = split_smd_sections(merged_mesh)

    legs_lines: List[str] = []
    torso_lines: List[str] = []
    legs_meshes: Set[str] = set()
    torso_meshes: Set[str] = set()
    n_legs = n_torso = n_both = 0

    i = 0
    while i < len(tri):
        mat = tri[i]
        if not smd_is_material_line(mat):
            raise RuntimeError(
                f"split_mesh_smd: expected material line at triangle {i}: {mat.strip()!r}")
        if i + 4 > len(tri):
            raise RuntimeError("split_mesh_smd: truncated triangle block")
        verts = tri[i + 1:i + 4]
        legs_votes = sum(1 for v in verts if smd_vertex_primary_bone(v) < splitbone)

        mat_name = mat.strip()
        if legs_votes == 3:
            legs_lines += [mat] + verts
            legs_meshes.add(mat_name)
            n_legs += 1
        elif legs_votes == 0:
            torso_lines += [mat] + verts
            torso_meshes.add(mat_name)
            n_torso += 1
        else:
            legs_lines += [mat] + verts
            torso_lines += [mat] + verts
            legs_meshes.add(mat_name)
            torso_meshes.add(mat_name)
            n_both += 1
        i += 4

    if n_legs == 0 or n_torso == 0:
        raise RuntimeError(
            f"split_mesh_smd: degenerate split (legs {n_legs}, torso {n_torso}) "
            f"— wrong splitbone {splitbone}?")

    legs_out.write_text("".join(header) + "".join(legs_lines) + "end\n",
                        encoding="utf-8", errors="ignore")
    torso_out.write_text("".join(header) + "".join(torso_lines) + "end\n",
                         encoding="utf-8", errors="ignore")
    print(f"== split-body: {n_legs} legs / {n_torso} torso / {n_both} overlap "
          f"triangle(s) at splitbone {splitbone} ==")
    return legs_meshes, torso_meshes


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
      <model>.iqm.bodyinfo
      <model>.iqm_<N>.skin
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

    for sidecar_suffix in (".framegroups", ".bodyinfo", ".events"):
        sc = iqm_path.with_name(iqm_path.name + sidecar_suffix)
        if sc.exists():
            shutil.move(str(sc), models_dir / sc.name)

    for skin in sorted(iqm_path.parent.glob(iqm_path.name + "_*.skin")):
        shutil.move(str(skin), models_dir / skin.name)

    return new_iqm, new_iqm_fg


def numbered_iqm_outputs(base_path: Path, count: int) -> List[Path]:
    """
    Returns:
      <base>01.iqm
      <base>02.iqm
      ...

    The suffix is at least two digits, growing only if there are 100+ meshes.
    """
    width = max(2, len(str(count)))
    return [
        base_path.with_name(f"{base_path.stem}{i:0{width}d}{base_path.suffix}")
        for i in range(1, count + 1)
    ]


def build_iqm(
    iqm_exe: Path,
    rotate: str,
    out_path: Path,
    mesh_paths: List[Path],
    anim_paths: List[Path],
    workdir: Path,
    modelname: str,
) -> None:
    if len(mesh_paths) > 1:
        print("== Merging multiple mesh SMDs into one ==")
        merged_mesh = workdir / f"{modelname}__merged_mesh.smd"
        merge_mesh_smds(mesh_paths, merged_mesh, strict=False)
        actual_mesh_paths = [merged_mesh]
    else:
        actual_mesh_paths = mesh_paths

    cmd = [str(iqm_exe), "-r", rotate, str(out_path)]
    cmd += [str(p) for p in actual_mesh_paths]
    cmd += [str(p) for p in anim_paths]

    run_cmd(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Half-Life MDL -> IQM pipeline (mdldec + ImageMagick + iqm) with QC-driven sequences.")
    ap.add_argument("mdl", help="Path to input .mdl")

    ap.add_argument("--mdldec", default=None, help="Path to mdldec.exe (default: next to this script)")
    ap.add_argument("--magick", default="magick.exe", help="Path to magick.exe (default: magick.exe from PATH)")
    ap.add_argument("--iqm", default=None, help="Path to iqm tool (default: next to this script)")

    ap.add_argument("--workdir", default="./work", help="Working directory (default: ./work)")
    ap.add_argument("--outdir", default=".", help="Output directory for IQM + framegroups (default: .)")
    ap.add_argument("--out", default=None, help="Explicit output IQM path (default: <outdir>/<mdlname>.iqm)")

    ap.add_argument("--rotate", default="0,0,90", help="iqm rotation '-r x,y,z' (default: 0,0,90)")
    ap.add_argument("--offset", default="0,0,0",
                    help="Root-bone offset 'x,y,z' applied to bone 0 in every SMD frame "
                         "(shifts entire skeleton+mesh; default: 0,0,0). Applied in source-MDL "
                         "coordinates BEFORE the iqm '-r rotate'. For HL models with the default "
                         "rotate '0,0,90', the Z axis stays Z.")
    ap.add_argument("--feet-z", type=float, default=None,
                    help="Target Z (final engine space) for the model's animated standing "
                         "feet, e.g. -24 for the Quake player hull. After the first build the "
                         "idle-frame feet height is measured by skinning the IQM; the root "
                         "bone is shifted by the difference and the IQM rebuilt. Overrides "
                         "guesswork --offset Z constants.")
    ap.add_argument("--split-body", action="store_true",
                    help="Additionally emit <name>_legs.iqm and <name>_torso.iqm: meshes "
                         "partitioned per-triangle at the detected splitbone (dominant "
                         "vertex weight), full skeleton + all anims in both, boundary "
                         "triangles duplicated into both halves (seam-hiding overlap ring). "
                         "For the DP/Zircon two-entity player split.")
    ap.add_argument("--keep-work", action="store_true", help="Do not wipe workdir before running")
    ap.add_argument("--delete-original-textures", action="store_true", help="After PNG conversion, delete non-PNG extracted textures")

    ap.add_argument("--qc", default=None, help="Explicit QC path (otherwise first .qc in workdir)")
    ap.add_argument("--material-prefix", default="textures/models", help="Material prefix in SMD (default: textures/models)")

    # Texture staging
    ap.add_argument("--content-root", default=None, help="If set, copy textures to <content-root>/textures/models/<modelname>/")
    ap.add_argument("--no-copy-textures", action="store_true", help="Disable copying textures even if --content-root is set")

    # Root motion
    ap.add_argument("--inplace", action="store_true",
                    help="DEPRECATED: blanket root-XY lock on ALL animations (old behavior). "
                         "Default is now per-sequence linear-motion extraction driven by the "
                         "MDL's STUDIO_LX/LY/LZ flags, which leaves non-locomotion anims intact.")
    ap.add_argument("--no-motion-extraction", action="store_true",
                    help="Disable per-sequence root-motion extraction entirely.")
    ap.add_argument("--no-blend-flatten", action="store_true",
                    help="Disable baking blended sequences at blend value 0 "
                         "(old behavior: first blend anim only = full-down aim).")

    # Bodygroups
    ap.add_argument("--no-bodygroups", action="store_true",
                    help="Ignore extra bodygroup submodels (old behavior: first studio of "
                         "each group only, no skin files).")

    # Multi-mesh support
    ap.add_argument("--split-per-submodel", action="store_true",
                    help="Emit one numbered IQM per submodel (<name>01.iqm, <name>02.iqm, ...) "
                         "instead of a single IQM containing every submodel as a mesh.")

    args = ap.parse_args()

    mdl_path = Path(args.mdl).resolve()
    if not mdl_path.exists():
        print(f"ERROR: MDL not found: {mdl_path}", file=sys.stderr)
        return 2

    workdir = Path(args.workdir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out).resolve() if args.out else (outdir / f"{mdl_path.stem}.iqm")

    # Python 3.12+ no longer resolves bare exe names against the cwd on
    # Windows — default to the copies shipped next to this script.
    script_dir = Path(__file__).resolve().parent
    mdldec_exe = Path(args.mdldec) if args.mdldec else script_dir / "mdldec.exe"
    magick_exe = Path(args.magick)
    iqm_exe = Path(args.iqm) if args.iqm else script_dir / "iqm.exe"

    modelname = mdl_path.stem

    print(f"== MDL:      {mdl_path}")
    print(f"== Workdir:  {workdir}")
    print(f"== Out IQM:  {out_path}")

    # 0) Parse MDL binary metadata (sequences motion flags, bodygroups, skin families)
    info = mdlinfo.parse_mdl(mdl_path)
    print(f"== MDL meta: {info.total_body_combos} body combo(s), "
          f"{info.numskinfamilies} skin family(ies), {len(info.sequences)} sequence(s)")

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

    # 4) Parse QC for bodygroups + sequences
    qc_path = Path(args.qc).resolve() if args.qc else find_first_qc(workdir)
    print(f"== Using QC: {qc_path}")

    groups, sequences = parse_qc(qc_path)
    if not groups:
        raise RuntimeError("No mesh SMDs found in QC ($body/$bodygroup).")
    if not sequences:
        print("!! Warning: No $sequence entries found in QC (model may be static).")

    bodygroup_mode = (not args.no_bodygroups and not args.split_per_submodel
                      and (info.total_body_combos > 1 or info.numskinfamilies > 1))

    if args.no_bodygroups or args.split_per_submodel:
        # old behavior: first (non-blank) studio per group
        for g in groups:
            g.studios = [s for s in g.studios if s][:1]

    # Flatten submodels; dedupe stems (a stem shared between groups would be
    # ambiguous — keep first, warn)
    submodels: List[SubmodelMeshes] = []
    seen_stems: Set[str] = set()
    for gi, g in enumerate(groups):
        for si, stem in enumerate(g.studios):
            if stem and stem.lower() in seen_stems:
                print(f"!! Duplicate submodel stem '{stem}' (group {g.name}); reusing first occurrence's meshes")
            if stem:
                seen_stems.add(stem.lower())
            submodels.append(SubmodelMeshes(group_index=gi, submodel_index=si, stem=stem))

    mesh_paths_by_stem: Dict[str, Path] = {}
    for sm in submodels:
        if sm.stem and sm.stem.lower() not in mesh_paths_by_stem:
            mesh_paths_by_stem[sm.stem.lower()] = resolve_file_recursive(workdir, sm.stem + ".smd")
    mesh_paths = list(mesh_paths_by_stem.values())

    # Legs/torso boundary for the QC upper/lower body split (emitted into
    # .bodyinfo; also gates --split-body)
    splitbone = detect_split_bone(parse_smd_nodes(mesh_paths[0])) if mesh_paths else None

    # Resolve anim SMDs; flatten multi-blend sequences to a single anim baked
    # at blend value 0deg (HLMV behavior) unless disabled.
    anim_paths: List[Path] = []
    flattened = 0
    for seq in sequences:
        stems = seq.blend_smds or [seq.smd_base]
        paths = [resolve_file_recursive(workdir, stem + ".smd") for stem in stems]
        if len(paths) > 1 and not args.no_blend_flatten:
            anim_paths.append(flatten_blend_sequence(seq, paths, workdir))
            flattened += 1
        else:
            anim_paths.append(paths[0])
    if flattened:
        print(f"== Flattened {flattened} blended sequence(s) at blend value 0 ==")

    print(f"== Meshes: {len(mesh_paths)} (bodygroup mode: {'ON' if bodygroup_mode else 'off'})")
    for p in mesh_paths:
        print("   -", p.name)
    print(f"== Sequences: {len(sequences)}")
    if sequences:
        animlist_path = workdir / "animlist.txt"
        animlist_path.write_text(",".join(seq.qc_name for seq in sequences), encoding="utf-8")
        print(f"== Wrote: {animlist_path.name} (debug)")

    # 5) Rewrite materials in ALL mesh SMDs
    print("== Rewriting SMD material names ==")
    stems_needed: Set[str] = set()
    total_replaced = 0
    for sm in submodels:
        if not sm.stem:
            continue
        mp = mesh_paths_by_stem[sm.stem.lower()]
        # b<group>s<submodel> tag: unique, compact, engine-safe
        tag = f"b{sm.group_index}s{sm.submodel_index}" if bodygroup_mode else ""
        replaced, stems, mesh_list = rewrite_smd_materials(
            mp, modelname, prefix=args.material_prefix, mesh_tag=tag)
        sm.meshes = mesh_list
        total_replaced += replaced
        stems_needed |= stems

    # In bodygroup mode, also stage every skin-family texture variant (skin
    # files reference them by plain stem)
    if bodygroup_mode:
        for fam in range(1, info.numskinfamilies):
            stems_needed |= {v.lower() for v in info.family_texture_map(fam).values()}

    # Anim SMDs usually don't have triangles; but safe to run (will be 0 most of the time)
    for apath in set(anim_paths):  # de-dupe identical paths
        replaced, stems, _ = rewrite_smd_materials(apath, modelname, prefix=args.material_prefix)
        total_replaced += replaced
        stems_needed |= stems

    print(f"== Material lines rewritten: {total_replaced} ==")

    # 5b) Root motion
    if args.inplace:
        print("== [DEPRECATED --inplace] blanket root-XY strip on all animations ==")
        for anim_path in set(anim_paths):
            strip_root_xy(anim_path)
    elif not args.no_motion_extraction:
        print("== Per-sequence root-motion extraction (STUDIO_LX/LY/LZ) ==")
        # Gate on the QC line's LX/LY/LZ tokens first (mdldec re-emits them and
        # they survive its label de-duping, e.g. binary "run"x3 -> run_1..3);
        # fall back to matching the MDL binary label. Axes are measured, so
        # the union set only gates.
        axes_by_path: Dict[Path, Set[int]] = {}
        for seq, apath in zip(sequences, anim_paths):
            s = info.sequence_by_label(seq.qc_name)
            if seq.has_motion or (s and s.motion_axes):
                axes = set(s.motion_axes) if (s and s.motion_axes) else {0, 1, 2}
                axes_by_path.setdefault(apath, set()).update(axes)
        for apath, axes in axes_by_path.items():
            extract_linear_motion(apath, axes)
        if not axes_by_path:
            print("   (no motion-flagged sequences)")

    # 5c) Root-bone offset — shifts the entire skeleton (and weighted mesh)
    # by a constant. Applied to mesh SMDs (reference pose) and anim SMDs
    # (every frame) so skeleton stays consistent across both.
    if args.offset and args.offset != "0,0,0":
        parts = args.offset.split(",")
        if len(parts) != 3:
            raise RuntimeError(f"--offset must be 'x,y,z', got: {args.offset}")
        dx, dy, dz = float(parts[0]), float(parts[1]), float(parts[2])
        print(f"== Offsetting root bone by ({dx}, {dy}, {dz}) ==")
        for mesh_path in set(mesh_paths):
            offset_root_bone(mesh_path, dx, dy, dz)
        for anim_path in set(anim_paths):
            offset_root_bone(anim_path, dx, dy, dz)

    # 6) Copy textures into content root (optional)
    if args.content_root and not args.no_copy_textures:
        content_root = Path(args.content_root).resolve()
        print(f"== Staging textures to content root: {content_root} ==")
        if bodygroup_mode:
            # stage under sanitized names — the .skin files reference these
            copies = [(sanitize_mesh_token(st), st) for st in sorted(stems_needed)]
        else:
            # underscore spaces to match the rewritten material paths
            copies = [(re.sub(r"\s+", "_", st), st) for st in sorted(stems_needed)]
        stage_textures_as_png(workdir, content_root, modelname, set(), copies)

    # 7) Generate framegroups data (based on QC order + anim SMD frame counts)
    framegroup_entries: List[Tuple[int, int, int, int, str]] = []
    if sequences:
        print("== Generating framegroups data ==")
        start = 0

        for seq, anim_path in zip(sequences, anim_paths):
            frames = count_smd_frames(anim_path)
            framegroup_entries.append((start, frames, seq.fps, seq.loop, seq.qc_name))
            start += frames
    else:
        print("== Skipping framegroups data (no sequences) ==")

    # 8) Run iqm tool
    print("== Building IQM ==")
    split_per_submodel = args.split_per_submodel and len(mesh_paths) > 1
    built_iqms: List[Path] = []

    if split_per_submodel:
        split_paths = numbered_iqm_outputs(out_path, len(mesh_paths))
        print(f"== Splitting into {len(split_paths)} IQM files (one per submodel) ==")

        for mesh_path, split_path in zip(mesh_paths, split_paths):
            print(f"== Building split IQM: {split_path.name} from {mesh_path.name} ==")
            if framegroup_entries:
                framegroups_path = split_path.with_suffix(split_path.suffix + ".framegroups")
                write_framegroups(framegroups_path, framegroup_entries)
                print(f"== Wrote: {framegroups_path} ==")
                events_path = split_path.with_suffix(split_path.suffix + ".events")
                if write_events(events_path, framegroup_entries, info):
                    print(f"== Wrote: {events_path} ==")

            build_iqm(
                iqm_exe=iqm_exe,
                rotate=args.rotate,
                out_path=split_path,
                mesh_paths=[mesh_path],
                anim_paths=anim_paths,
                workdir=workdir,
                modelname=modelname,
            )
            built_iqms.append(split_path)
    else:
        if framegroup_entries:
            framegroups_path = out_path.with_suffix(out_path.suffix + ".framegroups")  # <name>.iqm.framegroups
            write_framegroups(framegroups_path, framegroup_entries)
            print(f"== Wrote: {framegroups_path} ==")
            events_path = out_path.with_suffix(out_path.suffix + ".events")  # <name>.iqm.events
            if write_events(events_path, framegroup_entries, info):
                print(f"== Wrote: {events_path} ==")

        build_iqm(
            iqm_exe=iqm_exe,
            rotate=args.rotate,
            out_path=out_path,
            mesh_paths=mesh_paths,
            anim_paths=anim_paths,
            workdir=workdir,
            modelname=modelname,
        )
        built_iqms.append(out_path)

        # 8a) Measured feet-Z correction: skin the built IQM at the idle
        # frame, shift the root by the miss, rebuild once. Runs BEFORE the
        # body-half split so the halves inherit the corrected offset.
        if args.feet_z is not None:
            rot = [float(v) for v in args.rotate.split(",")]
            if rot[0] != 0.0 or rot[1] != 0.0:
                print("!! --feet-z assumes a Z-preserving --rotate (yaw only); "
                      f"got {args.rotate} — measurement may be wrong")
            idle_frame = 0
            for (fg_start, _c, _f, _l, fg_name) in framegroup_entries:
                if fg_name == "idle":
                    idle_frame = fg_start
                    break
            measured = measure_animated_feet_z(out_path, idle_frame)
            if abs(measured - args.feet_z) > 0.5:
                dzz = args.feet_z - measured
                print(f"== feet-z: measured {measured:.2f}, target {args.feet_z:g} "
                      f"-> shifting anim roots by {dzz:+.2f} and rebuilding ==")
                # ANIM SMDs only. Skinning is animPose * inverse(bindPose):
                # shifting the mesh (bind) skeleton along with the anims
                # cancels out exactly — the historical --offset 0,0,24 was a
                # no-op for on-screen placement for this very reason.
                for anim_path in set(anim_paths):
                    offset_root_bone(anim_path, 0.0, 0.0, dzz)
                build_iqm(
                    iqm_exe=iqm_exe,
                    rotate=args.rotate,
                    out_path=out_path,
                    mesh_paths=mesh_paths,
                    anim_paths=anim_paths,
                    workdir=workdir,
                    modelname=modelname,
                )
                remeasured = measure_animated_feet_z(out_path, idle_frame)
                print(f"== feet z: {measured:.2f} -> {remeasured:.2f} "
                      f"(target {args.feet_z:g}) ==")
                if abs(remeasured - args.feet_z) > 0.5:
                    raise RuntimeError(
                        f"--feet-z correction missed (got {remeasured:.2f}, "
                        f"wanted {args.feet_z:g}) — feet not root-driven?")
            else:
                print(f"== feet z: {measured:.2f} already at target {args.feet_z:g} ==")

        # 8b) Bodygroup skin files + bodyinfo sidecar
        bases = compute_group_bases(groups, info)
        if bodygroup_mode:
            write_skin_files(out_path, groups, bases, submodels, info,
                             modelname, args.material_prefix)
            write_bodyinfo(out_path, groups, bases, info, sequences, splitbone)
        elif splitbone is not None:
            # minimal sidecar: QC needs the splitbone even without bodygroups
            write_bodyinfo(out_path, groups, bases, info, sequences, splitbone)

        # 8c) Body-half split (DP/Zircon two-entity player)
        if args.split_body:
            if splitbone is None:
                print("!! --split-body requested but no splitbone detected; skipping halves")
            else:
                print("== Building body-half IQMs (--split-body) ==")
                split_src = workdir / f"{modelname}__splitsrc.smd"
                merge_mesh_smds(mesh_paths, split_src, strict=False)
                legs_smd = workdir / f"{modelname}__legs_mesh.smd"
                torso_smd = workdir / f"{modelname}__torso_mesh.smd"
                legs_meshes, torso_meshes = split_mesh_smd(
                    split_src, splitbone[0], legs_smd, torso_smd)

                halves = (
                    (out_path.with_name(f"{out_path.stem}_legs{out_path.suffix}"),
                     legs_smd, legs_meshes),
                    (out_path.with_name(f"{out_path.stem}_torso{out_path.suffix}"),
                     torso_smd, torso_meshes),
                )
                for half_path, half_smd, half_meshes in halves:
                    if framegroup_entries:
                        write_framegroups(
                            half_path.with_suffix(half_path.suffix + ".framegroups"),
                            framegroup_entries)
                        write_events(
                            half_path.with_suffix(half_path.suffix + ".events"),
                            framegroup_entries, info)
                    build_iqm(
                        iqm_exe=iqm_exe,
                        rotate=args.rotate,
                        out_path=half_path,
                        mesh_paths=[half_smd],
                        anim_paths=anim_paths,
                        workdir=workdir,
                        modelname=modelname,
                    )
                    if bodygroup_mode:
                        write_skin_files(half_path, groups, bases, submodels, info,
                                         modelname, args.material_prefix,
                                         mesh_filter=half_meshes)
                    write_bodyinfo(half_path, groups, bases, info, sequences, splitbone)
                    built_iqms.append(half_path)

    # 9) Move final outputs into content root (optional)
    if args.content_root:
        content_root = Path(args.content_root).resolve()
        print(f"== Moving outputs to content root: {content_root} ==")

        for built_iqm in built_iqms:
            new_iqm, new_fg = move_outputs_to_content_root(built_iqm, content_root, modelname)

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
