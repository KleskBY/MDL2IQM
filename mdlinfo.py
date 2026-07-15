#!/usr/bin/env python3
"""
Half-Life MDL binary metadata parser.

Reads the studiohdr_t tables that the QC decompile loses or mangles:
  - sequences: motiontype (STUDIO_LX/LY/LZ) + linearmovement for proper
    per-sequence root-motion extraction
  - bodyparts: group names, submodel counts, HL 'base' body encoding,
    submodel names
  - textures + skin families: per-family texture remap tables

Struct layouts from halflife_src/engine/studio.h (design reference only,
never compiled).
"""
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# motiontype flags (studio.h:338)
STUDIO_X = 0x0001
STUDIO_Y = 0x0002
STUDIO_Z = 0x0004
STUDIO_LX = 0x0040
STUDIO_LY = 0x0080
STUDIO_LZ = 0x0100

# sequence flags (studio.h:357)
STUDIO_LOOPING = 0x0001

# texture flags (studio.h:329)
STUDIO_NF_FLATSHADE = 0x0001
STUDIO_NF_CHROME = 0x0002
STUDIO_NF_FULLBRIGHT = 0x0004
STUDIO_NF_NOMIPS = 0x0008
STUDIO_NF_ALPHA = 0x0010
STUDIO_NF_ADDITIVE = 0x0020
STUDIO_NF_MASKED = 0x0040


def _cstr(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].decode("latin1")


@dataclass
class MdlSequence:
    label: str
    fps: float
    looping: bool
    numframes: int
    motiontype: int
    linearmovement: tuple  # (x, y, z) total root displacement over sequence

    @property
    def motion_axes(self) -> List[int]:
        """Axis indices (0=x,1=y,2=z) flagged for linear motion extraction."""
        axes = []
        if self.motiontype & STUDIO_LX:
            axes.append(0)
        if self.motiontype & STUDIO_LY:
            axes.append(1)
        if self.motiontype & STUDIO_LZ:
            axes.append(2)
        return axes


@dataclass
class MdlSubmodel:
    name: str  # mstudiomodel_t.name — matches mdldec SMD stem


@dataclass
class MdlBodypart:
    name: str
    base: int  # HL body encoding: submodel index = (body / base) % nummodels
    models: List[MdlSubmodel] = field(default_factory=list)


@dataclass
class MdlTexture:
    name: str  # e.g. "sci_head.bmp"
    flags: int


@dataclass
class MdlInfo:
    name: str
    sequences: List[MdlSequence] = field(default_factory=list)
    bodyparts: List[MdlBodypart] = field(default_factory=list)
    textures: List[MdlTexture] = field(default_factory=list)
    numskinref: int = 0
    numskinfamilies: int = 1
    skin_table: List[List[int]] = field(default_factory=list)  # [family][skinref] -> texture idx

    @property
    def total_body_combos(self) -> int:
        n = 1
        for bp in self.bodyparts:
            n *= max(1, len(bp.models))
        return n

    def sequence_by_label(self, label: str) -> "MdlSequence":
        for s in self.sequences:
            if s.label.lower() == label.lower():
                return s
        return None

    def family_texture_map(self, family: int) -> Dict[str, str]:
        """
        Map family-0 texture stem (lowercase) -> family-N texture stem, for
        every skinref slot that differs. mdldec writes family-0 texture names
        as SMD materials, so this remaps those stems for other families.
        """
        remap: Dict[str, str] = {}
        if family <= 0 or family >= len(self.skin_table):
            return remap
        for ref in range(self.numskinref):
            t0 = self.skin_table[0][ref]
            tf = self.skin_table[family][ref]
            if t0 != tf and t0 < len(self.textures) and tf < len(self.textures):
                stem0 = Path(self.textures[t0].name).stem.lower()
                stemf = Path(self.textures[tf].name).stem
                remap[stem0] = stemf
        return remap


# studiohdr_t (studio.h:47): id, version, name[64], length,
# eyeposition/min/max/bbmin/bbmax (15 floats), flags, then 26 index ints.
_HDR = struct.Struct("<2i64si15f27i")
# mstudioseqdesc_t (studio.h:164), 176 bytes
_SEQ = struct.Struct("<32sf10i3f2i6f2i2i2f2f6i")
# mstudiobodyparts_t (studio.h:257), 76 bytes
_BODYPART = struct.Struct("<64s3i")
# mstudiomodel_t (studio.h:282), 112 bytes
_MODEL = struct.Struct("<64sif10i")
# mstudiotexture_t (studio.h:268), 80 bytes
_TEXTURE = struct.Struct("<64s4i")


def parse_mdl(path: Path, _allow_tfile: bool = True) -> MdlInfo:
    data = Path(path).read_bytes()

    hdr = _HDR.unpack_from(data, 0)
    magic, version = hdr[0], hdr[1]
    if magic != int.from_bytes(b"IDST", "little"):
        raise ValueError(f"Not a studio MDL (bad magic): {path}")
    if version != 10:
        raise ValueError(f"Unsupported MDL version {version} (want 10): {path}")

    name = _cstr(hdr[2])
    ints = hdr[19:]  # 27 ints starting at 'flags'
    (flags,
     numbones, boneindex,
     numbonecontrollers, bonecontrollerindex,
     numhitboxes, hitboxindex,
     numseq, seqindex,
     numseqgroups, seqgroupindex,
     numtextures, textureindex, texturedataindex,
     numskinref, numskinfamilies, skinindex,
     numbodyparts, bodypartindex,
     numattachments, attachmentindex,
     soundtable, soundindex, soundgroups, soundgroupindex,
     numtransitions, transitionindex) = ints

    info = MdlInfo(name=name, numskinref=numskinref,
                   numskinfamilies=max(1, numskinfamilies))

    for i in range(numseq):
        s = _SEQ.unpack_from(data, seqindex + i * _SEQ.size)
        label = _cstr(s[0])
        fps = s[1]
        seqflags = s[2]
        # after fps: flags activity actweight numevents eventindex numframes
        #            numpivots pivotindex motiontype motionbone
        numframes = s[7]
        motiontype = s[10]
        linearmovement = (s[12], s[13], s[14])
        info.sequences.append(MdlSequence(
            label=label, fps=fps, looping=bool(seqflags & STUDIO_LOOPING),
            numframes=numframes, motiontype=motiontype,
            linearmovement=linearmovement))

    for i in range(numbodyparts):
        bname, nummodels, base, modelindex = _BODYPART.unpack_from(
            data, bodypartindex + i * _BODYPART.size)
        bp = MdlBodypart(name=_cstr(bname), base=base)
        for j in range(nummodels):
            m = _MODEL.unpack_from(data, modelindex + j * _MODEL.size)
            bp.models.append(MdlSubmodel(name=_cstr(m[0])))
        info.bodyparts.append(bp)

    for i in range(numtextures):
        t = _TEXTURE.unpack_from(data, textureindex + i * _TEXTURE.size)
        info.textures.append(MdlTexture(name=_cstr(t[0]), flags=t[1]))

    for f in range(info.numskinfamilies):
        row_ofs = skinindex + f * numskinref * 2
        row = list(struct.unpack_from(f"<{numskinref}h", data, row_ofs))
        info.skin_table.append(row)

    # $externaltextures models keep textures + skin families in a sibling
    # "<name>t.mdl" (GoldSrc convention). Pull those tables in.
    if _allow_tfile and numtextures == 0:
        p = Path(path)
        for suffix in ("t.mdl", "T.mdl"):
            tpath = p.with_name(p.stem + suffix)
            if tpath.exists():
                tinfo = parse_mdl(tpath, _allow_tfile=False)
                info.textures = tinfo.textures
                info.numskinref = tinfo.numskinref
                info.numskinfamilies = tinfo.numskinfamilies
                info.skin_table = tinfo.skin_table
                break

    return info


if __name__ == "__main__":
    import sys
    info = parse_mdl(Path(sys.argv[1]))
    print(f"model: {info.name}")
    print(f"skin families: {info.numskinfamilies}  skinrefs: {info.numskinref}")
    print(f"body combos: {info.total_body_combos}")
    for bp in info.bodyparts:
        print(f"  bodypart '{bp.name}' base={bp.base}:")
        for m in bp.models:
            print(f"    - {m.name}")
    print("textures:")
    for t in info.textures:
        print(f"  {t.name} flags=0x{t.flags:x}")
    print("sequences:")
    for s in info.sequences:
        lm = s.linearmovement
        motion = "".join(a for a, bit in (("LX", STUDIO_LX), ("LY", STUDIO_LY), ("LZ", STUDIO_LZ))
                         if s.motiontype & bit) or "-"
        print(f"  {s.label:24s} frames={s.numframes:4d} fps={s.fps:5.1f} "
              f"loop={int(s.looping)} motion={motion} lm=({lm[0]:.1f},{lm[1]:.1f},{lm[2]:.1f})")
