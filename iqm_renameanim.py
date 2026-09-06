#!/usr/bin/env python3
"""Rename an animation sequence in place, everywhere its name appears.

Some converted rigs don't carry the exact sequence names the game code asks
for (e.g. a "run2"/"walk2handed" pair instead of plain "run"/"walk" -- see
STALKER_Marked1, whose source model has no bare two-handed-agnostic gait
cycle). `Player_AnimResolve` (src/serverqc/player/animation.qc) looks the name
up two ways depending on engine: FTE's `frameforname` reads the name straight
out of the IQM's own anim table, DarkPlaces-class engines read the
`.framegroups` sidecar instead. Both have to agree, so both get patched.

The IQM patch reuses the exact same trick as iqm_fixmat.py: IQM strings live
in one null-terminated blob addressed by offset, so writing the new (shorter
or equal-length) name plus its terminator over the old one leaves every other
offset valid, provided the new name is not longer than the old one -- pad with
a distinguishing suffix on the OLD name at conversion time if a same-length
swap isn't available, rather than trying to grow the blob here.

    python tools/mdl2iqm/iqm_renameanim.py models/player/STALKER_Marked1.iqm --rename run2=run --rename walk2handed=walk
    python tools/mdl2iqm/iqm_renameanim.py models/player/*.iqm --rename run2=run --dry-run
"""

import argparse
import glob
import os
import struct
import sys


def parse_pairs(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"expected OLD=NEW, got {p!r}")
        old, new = p.split("=", 1)
        out[old] = new
    return out


def rename_in_iqm(path, renames, dry):
    with open(path, "rb") as fh:
        d = bytearray(fh.read())
    if not bytes(d[:16]).startswith(b"INTERQUAKEMODEL"):
        print("  %s: not an IQM" % path)
        return 0

    v = struct.unpack("<27I", bytes(d[16:16 + 108]))
    num_text, ofs_text = v[3], v[4]
    num_anims, ofs_anims = v[17], v[18]

    def cstr(off):
        end = d.index(0, ofs_text + off)
        return bytes(d[ofs_text + off:end]).decode("latin1")

    fixed = 0
    off = ofs_anims
    for _ in range(num_anims):
        name_off, = struct.unpack("<I", bytes(d[off:off + 4]))
        off += 20  # name, first_frame, num_frames, framerate, flags

        # iqm.exe names each anim after the full SMD staging path it was
        # invoked with (e.g. "...\work\run2"), not a bare "run2" -- same in
        # the already-shipping player.iqm. Match/replace only the final path
        # component, so the prefix stays whatever convention the engine's
        # own lookup already relies on.
        old = cstr(name_off)
        sep = max(old.rfind("/"), old.rfind("\\"))
        prefix, old_base = old[:sep + 1], old[sep + 1:]
        new_base = renames.get(old_base)
        if new_base is None:
            continue
        new = prefix + new_base
        if len(new) > len(old):
            print(f"  !! {os.path.basename(path)}: {old_base!r} -> {new_base!r} is "
                  "longer than the original, skipping (IQM strings can't grow in place)")
            continue

        print("  %-22s %s -> %s" % (os.path.basename(path), old_base, new_base))
        fixed += 1
        if not dry:
            at = ofs_text + name_off
            d[at:at + len(new)] = new.encode("latin1")
            d[at + len(new)] = 0  # terminate early; the rest is padding

    if fixed and not dry:
        with open(path, "wb") as fh:
            fh.write(bytes(d))
    return fixed


def rename_in_framegroups(path, renames, dry):
    if not os.path.exists(path):
        return 0
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    fixed = 0
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5 and parts[4] in renames:
            new_name = renames[parts[4]]
            print("  %-22s (framegroups) %s -> %s"
                  % (os.path.basename(path), parts[4], new_name))
            parts[4] = new_name
            fixed += 1
        out.append(" ".join(parts))
    if fixed and not dry:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
    return fixed


def rename_in_bodyinfo(path, renames, dry):
    if not os.path.exists(path):
        return 0
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    fixed = 0
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "anim" and parts[1] in renames:
            new_name = renames[parts[1]]
            print("  %-22s (bodyinfo) anim %s -> %s"
                  % (os.path.basename(path), parts[1], new_name))
            parts[1] = new_name
            fixed += 1
        out.append(" ".join(parts))
    if fixed and not dry:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="IQM path(s)/glob(s)")
    ap.add_argument("--rename", action="append", required=True, metavar="OLD=NEW",
                    help="Sequence-name pair, applied to every model given. Repeatable.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        renames = parse_pairs(args.rename)
    except ValueError as e:
        ap.error(str(e))
        return 2

    paths = []
    for pat in args.models:
        paths.extend(sorted(glob.glob(pat)) or [pat])

    total = 0
    for p in paths:
        total += rename_in_iqm(p, renames, args.dry_run)
        total += rename_in_framegroups(p + ".framegroups", renames, args.dry_run)
        total += rename_in_bodyinfo(p + ".bodyinfo", renames, args.dry_run)

    print("%d rename(s) %s across %d file(s)"
          % (total, "would be applied" if args.dry_run else "applied", len(paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
