#!/usr/bin/env python3
"""Rebuild an HL model so it keeps its own mesh and skeleton but plays a
different model's animations.

Why: the game's player animation code (src/serverqc/player/animation.qc)
resolves poses BY NAME, against the stock Half-Life player.mdl sequence list
(idle, deep_idle, walk, walk2handed, run, run2, crouch_idle, die_simple,
ref_aim_<weapon>, crouch_shoot_<weapon>, ...). A model rigged for a different
game names its sequences differently and is missing whole categories, so those
lookups return -1, the frame never updates, and the model renders its bind pose
(reads in game as a T-pose). Patching that up with per-name aliases only ever
approximates it, and invents poses that do not exist on the source rig at all.

Retargeting instead gives such a model the real animset. Two HL biped rigs are
structurally 1:1 even when they disagree on names (Bip01 R Arm1 vs
Bip01 R UpperArm, Bip01 L Leg vs Bip01 L Thigh), bone lengths and per-bone rest
rotations, because both put the child offset down local +X. So: take each
bone's WORLD orientation from the source animation, recompute the local
rotation against the target's own parent chain, and keep the target's own bone
offsets. Rest-pose and proportion differences cancel out; the target strikes
exactly the source's pose with its own proportions. Measured bone-direction
error doing this HL -> CS1.6 is 0.0000 degrees.

    python tools/mdl2iqm/hl_retarget.py --anim-mdl exoskeleton_dolg.mdl \
                                        --mesh-mdl dolg.mdl --out dolg.mdl

The result is a normal HL .mdl; feed it to mdl2iqm.py as usual.
"""

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

MDL2IQM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MDL2IQM_DIR))
from mdl2iqm import parse_qc  # noqa: E402

# The same joint under two naming conventions. Exact name wins first, so only
# genuine disagreements need listing.
SYNONYMS = {
    "bip01 l arm": "bip01 l clavicle", "bip01 l arm1": "bip01 l upperarm",
    "bip01 l arm2": "bip01 l forearm",
    "bip01 r arm": "bip01 r clavicle", "bip01 r arm1": "bip01 r upperarm",
    "bip01 r arm2": "bip01 r forearm",
    "bip01 l leg": "bip01 l thigh", "bip01 l leg1": "bip01 l calf",
    "bip01 r leg": "bip01 r thigh", "bip01 r leg1": "bip01 r calf",
}


def mat_from_euler(rx, ry, rz):
    """R = Rz * Ry * Rx, matching studiomdl's AngleQuaternion."""
    ca, sa = math.cos(rx), math.sin(rx)
    cb, sb = math.cos(ry), math.sin(ry)
    cc, sc = math.cos(rz), math.sin(rz)
    return [
        [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
        [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
        [-sb,     cb * sa,                cb * ca],
    ]


def euler_from_mat(m):
    ry = math.asin(max(-1.0, min(1.0, -m[2][0])))
    if abs(m[2][0]) < 0.999999:
        return math.atan2(m[2][1], m[2][2]), ry, math.atan2(m[1][0], m[0][0])
    return math.atan2(-m[1][2], m[1][1]), ry, 0.0          # gimbal lock


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mt(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def parse_smd(path):
    """-> (nodes {id: (name, parent)}, order [id], frames [{id: [6 floats]}])"""
    nodes, order, frames = {}, [], []
    sec = None
    node_re = re.compile(r'^\s*(-?\d+)\s+"([^"]*)"\s+(-?\d+)\s*$')
    for raw in Path(path).read_text(encoding="latin1").splitlines():
        s = raw.strip()
        low = s.lower()
        if low in ("nodes", "skeleton", "triangles"):
            sec = low
            continue
        if low == "end":
            sec = None
            continue
        if sec == "nodes":
            m = node_re.match(raw)
            if m:
                bid = int(m.group(1))
                nodes[bid] = (m.group(2), int(m.group(3)))
                order.append(bid)
        elif sec == "skeleton":
            if low.startswith("time"):
                frames.append({})
                continue
            p = s.split()
            if len(p) >= 7 and frames:
                frames[-1][int(p[0])] = [float(x) for x in p[1:7]]
    return nodes, order, frames


def build_bone_map(src_nodes, dst_nodes):
    dst_by_name = {n.lower(): bid for bid, (n, _) in dst_nodes.items()}
    out = {}
    for sid, (name, _) in src_nodes.items():
        key = name.lower()
        did = dst_by_name.get(key)
        if did is None and key in SYNONYMS:
            did = dst_by_name.get(SYNONYMS[key])
        if did is not None:
            out[sid] = did
    return out


def world_pose(nodes, order, frame, bind):
    W = {}
    for bid in order:
        _, par = nodes[bid]
        v = frame.get(bid) or bind.get(bid) or [0] * 6
        R, T = mat_from_euler(v[3], v[4], v[5]), [v[0], v[1], v[2]]
        if par < 0:
            W[bid] = (R, T)
        else:
            PR, PT = W[par]
            rot = mv(PR, T)
            W[bid] = (mm(PR, R), [PT[i] + rot[i] for i in range(3)])
    return W


def retarget(src_smd, dst_bind_smd, out_smd):
    s_nodes, s_order, s_frames = parse_smd(src_smd)
    d_nodes, d_order, d_frames = parse_smd(dst_bind_smd)
    if not s_frames or not d_frames:
        raise RuntimeError(f"no frames in {src_smd} or {dst_bind_smd}")
    s_bind, d_bind = s_frames[0], d_frames[0]
    d2s = {v: k for k, v in build_bone_map(s_nodes, d_nodes).items()}

    out = ["version 1", "nodes"]
    out += [f'{bid} "{d_nodes[bid][0]}" {d_nodes[bid][1]}' for bid in d_order]
    out += ["end", "skeleton"]

    for fi, sframe in enumerate(s_frames):
        SW = world_pose(s_nodes, s_order, sframe, s_bind)
        out.append(f"time {fi}")
        DW = {}
        for bid in d_order:
            _, par = d_nodes[bid]
            bindv = d_bind.get(bid, [0] * 6)
            sid = d2s.get(bid)
            if sid is not None and sid in SW:
                # Root carries the source's animated translation so locomotion
                # survives; every other bone keeps the target's own offset, so
                # the target keeps its proportions. mdl2iqm's --feet-z pass
                # re-plants the feet afterwards.
                if par < 0:
                    R_local, T_local = SW[sid][0], SW[sid][1]
                else:
                    R_local = mm(mt(DW[par][0]), SW[sid][0])
                    T_local = [bindv[0], bindv[1], bindv[2]]
            else:
                R_local = mat_from_euler(bindv[3], bindv[4], bindv[5])
                T_local = [bindv[0], bindv[1], bindv[2]]
            if par < 0:
                DW[bid] = (R_local, T_local)
            else:
                PR, PT = DW[par]
                rot = mv(PR, T_local)
                DW[bid] = (mm(PR, R_local), [PT[i] + rot[i] for i in range(3)])
            rx, ry, rz = euler_from_mat(R_local)
            out.append(f"{bid} {T_local[0]:.6f} {T_local[1]:.6f} {T_local[2]:.6f} "
                       f"{rx:.6f} {ry:.6f} {rz:.6f}")
    out.append("end")
    Path(out_smd).write_text("\n".join(out) + "\n", encoding="latin1")
    return len(s_frames)


def max_bone_error(src_smd, dst_bind_smd, out_smd):
    """Worst world-space bone-direction disagreement, in degrees. 0 means the
    target reproduces the source pose exactly."""
    s_nodes, s_order, s_frames = parse_smd(src_smd)
    d_nodes, d_order, d_frames = parse_smd(dst_bind_smd)
    o_nodes, o_order, o_frames = parse_smd(out_smd)
    s2d = build_bone_map(s_nodes, d_nodes)
    worst = 0.0
    for fi in {0, len(s_frames) // 2, len(s_frames) - 1}:
        SW = world_pose(s_nodes, s_order, s_frames[fi], s_frames[0])
        OW = world_pose(o_nodes, o_order, o_frames[fi], d_frames[0])
        for sid, did in s2d.items():
            a, b = mv(SW[sid][0], [1, 0, 0]), mv(OW[did][0], [1, 0, 0])
            dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
            worst = max(worst, math.degrees(math.acos(dot)))
    return worst


def find_qc(d):
    qcs = sorted(Path(d).glob("*.qc"))
    if not qcs:
        raise SystemExit(f"no .qc found in {d}")
    return qcs[0]


def decompile(mdldec, mdl, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(mdldec), str(mdl), str(outdir)],
                       capture_output=True, text=True, errors="ignore")
    if r.returncode != 0:
        raise SystemExit(f"mdldec failed on {mdl}:\n{r.stdout}{r.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anim-mdl", required=True, help=".mdl supplying the ANIMATIONS")
    ap.add_argument("--mesh-mdl", required=True, help=".mdl supplying MESH + skeleton")
    ap.add_argument("--out", help="output .mdl name (default: <mesh>_retargeted.mdl)")
    ap.add_argument("--workdir", default=None, help="scratch dir (default: ./hl_retarget_work)")
    ap.add_argument("--mdldec", default=str(MDL2IQM_DIR / "mdldec.exe"))
    ap.add_argument("--studiomdl", default=str(MDL2IQM_DIR / "studiomdl.exe"))
    ap.add_argument("--verify", action="store_true",
                    help="report worst bone-direction error after retargeting")
    args = ap.parse_args()

    anim_mdl, mesh_mdl = Path(args.anim_mdl).resolve(), Path(args.mesh_mdl).resolve()
    work = Path(args.workdir).resolve() if args.workdir else Path.cwd() / "hl_retarget_work"
    anim_dir, mesh_dir = work / "anim", work / "mesh"
    out_name = args.out or f"{mesh_mdl.stem}_retargeted.mdl"

    print(f"== decompiling {anim_mdl.name} (animations)")
    decompile(args.mdldec, anim_mdl, anim_dir)
    print(f"== decompiling {mesh_mdl.name} (mesh + skeleton)")
    decompile(args.mdldec, mesh_mdl, mesh_dir)

    _, a_seqs = parse_qc(find_qc(anim_dir))
    m_qc = find_qc(mesh_dir)
    m_groups, _ = parse_qc(m_qc)
    if not m_groups:
        raise SystemExit("mesh model has no $body/$bodygroup")

    bind_smd = mesh_dir / f"{next(s for g in m_groups for s in g.studios if s)}.smd"
    print(f"== skeleton/bind from {bind_smd.name}")

    stems = {s for seq in a_seqs for s in (seq.blend_smds or [seq.smd_base])}
    print(f"== retargeting {len(stems)} SMD(s) across {len(a_seqs)} sequence(s)")
    worst, done = 0.0, 0
    for stem in sorted(stems):
        src = anim_dir / f"{stem}.smd"
        if not src.exists():
            print(f"   !! missing {src.name}, skipped")
            continue
        dst = mesh_dir / f"rt_{stem}.smd"
        retarget(src, bind_smd, dst)
        if args.verify:
            worst = max(worst, max_bone_error(src, bind_smd, dst))
        done += 1
    print(f"== retargeted {done} SMD(s)")
    if args.verify:
        print(f"== worst bone-direction error: {worst:.4f} deg")

    # Merged QC: the mesh model's own directives (bodies, bodygroups, textures)
    # with the animation model's sequences pointed at the retargeted SMDs.
    keep = [ln for ln in m_qc.read_text(encoding="latin1").splitlines()
            if not re.match(r"^\s*\$sequence", ln, re.I)]
    keep = [re.sub(r'^\s*\$modelname\s+".*"', f'$modelname "{out_name}"', ln, flags=re.I)
            for ln in keep]
    lines = keep + ["", f"// sequences retargeted from {anim_mdl.name}", ""]
    for seq in a_seqs:
        stems_l = seq.blend_smds or [seq.smd_base]
        tail = ""
        if len(stems_l) > 1:
            tail += f" blend XR {seq.blend_start:g} {seq.blend_end:g}"
        if seq.has_motion:
            tail += " LX"
        tail += f" fps {seq.fps}"
        if seq.loop:
            tail += " loop"
        refs = " ".join(f'"rt_{s}"' for s in stems_l)
        lines.append(f'$sequence "{seq.qc_name}" {refs}{tail}')

    qc_out = mesh_dir / "retargeted.qc"
    qc_out.write_text("\n".join(lines) + "\n", encoding="latin1")

    r = subprocess.run([args.studiomdl, qc_out.name], cwd=str(mesh_dir),
                       capture_output=True, text=True, errors="ignore")
    if r.returncode != 0:
        print((r.stdout or "") + (r.stderr or ""))
        raise SystemExit(f"studiomdl failed (exit {r.returncode})")

    built = mesh_dir / out_name
    if not built.exists():
        raise SystemExit(f"studiomdl reported success but {built} is missing")
    print(f"== built {built}  ({built.stat().st_size} bytes)")
    for extra in mesh_dir.glob(f"{built.stem}T.mdl"):
        print(f"== external textures: {extra.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
