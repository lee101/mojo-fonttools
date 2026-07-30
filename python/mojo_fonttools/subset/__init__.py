from __future__ import annotations

import math
import operator
import struct
from dataclasses import dataclass

from ..ttLib.sfnt import STANDARD_GLYPH_ORDER


class SubsetError(ValueError):
    pass


@dataclass(frozen=True)
class GlyfSubset:
    glyf: bytes
    loca: tuple[int, ...]
    glyph_ids: tuple[int, ...]
    old_to_new: dict[int, int]


def _component_records(data):
    if len(data) < 10 or struct.unpack_from(">h", data)[0] >= 0:
        return []
    pos, records, more = 10, [], True
    while more:
        if pos + 4 > len(data):
            raise SubsetError("truncated composite glyph")
        flags, gid = struct.unpack_from(">HH", data, pos)
        records.append((gid, pos + 2))
        pos += 4
        pos += 4 if flags & 0x0001 else 2
        if flags & 0x0008:
            pos += 2
        elif flags & 0x0040:
            pos += 4
        elif flags & 0x0080:
            pos += 8
        if pos > len(data):
            raise SubsetError("truncated composite glyph")
        more = bool(flags & 0x0020)
    return records


def subset_glyf(glyf, loca, glyph_ids, *, include_notdef=True):
    """Subset raw ``glyf``/``loca`` data and close over composite components."""
    data = bytes(glyf)
    try:
        offsets = tuple(operator.index(value) for value in loca)
        selected = {operator.index(gid) for gid in glyph_ids}
    except TypeError as error:
        raise SubsetError("loca offsets and glyph ids must be integers") from error
    if len(offsets) < 2 or offsets[0] != 0 or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise SubsetError("loca offsets must be sorted and start at zero")
    if offsets[-1] > len(data):
        raise SubsetError("loca offsets point outside glyf data")
    glyph_count = len(offsets) - 1
    if include_notdef and glyph_count:
        selected.add(0)
    if any(gid < 0 or gid >= glyph_count for gid in selected):
        raise SubsetError("glyph id outside loca range")
    pending = list(selected)
    while pending:
        gid = pending.pop()
        raw = data[offsets[gid] : offsets[gid + 1]]
        for component_gid, _ in _component_records(raw):
            if component_gid >= glyph_count:
                raise SubsetError(f"composite references invalid glyph {component_gid}")
            if component_gid not in selected:
                selected.add(component_gid)
                pending.append(component_gid)
    ordered = tuple(sorted(selected))
    remap = {old: new for new, old in enumerate(ordered)}
    output, new_loca = bytearray(), [0]
    for old_gid in ordered:
        raw = bytearray(data[offsets[old_gid] : offsets[old_gid + 1]])
        for component_gid, position in _component_records(raw):
            struct.pack_into(">H", raw, position, remap[component_gid])
        output.extend(raw)
        if len(output) & 1:
            output.append(0)
        new_loca.append(len(output))
    return GlyfSubset(bytes(output), tuple(new_loca), ordered, remap)


class Options:
    def __init__(self, **kwargs):
        self.notdef_glyph = True
        self.retain_gids = False
        self.hinting = True
        self.layout_features = []
        for name, value in kwargs.items():
            setattr(self, name, value)


class Subsetter:
    """A fontTools-compatible driver for TrueType outline subsetting."""

    def __init__(self, options=None):
        self.options = options or Options()
        self.glyph_names_requested = set()
        self.glyph_ids_requested = set()
        self.unicodes_requested = set()

    def populate(self, glyphs=(), gids=(), unicodes=(), text=""):
        self.glyph_names_requested.update(glyphs)
        try:
            self.glyph_ids_requested.update(operator.index(gid) for gid in gids)
            self.unicodes_requested.update(operator.index(value) for value in unicodes)
        except TypeError as error:
            raise SubsetError("glyph ids and Unicode values must be integers") from error
        self.unicodes_requested.update(map(ord, text))

    def subset(self, font):
        glyph_order = font.getGlyphOrder()
        reverse = font.getReverseGlyphMap()
        requested = set(self.glyph_ids_requested)
        for name in self.glyph_names_requested:
            if name not in reverse:
                raise SubsetError(f"unknown glyph {name!r}")
            requested.add(reverse[name])
        cmap = font.getBestCmap()
        for codepoint in self.unicodes_requested:
            name = cmap.get(codepoint)
            if name is not None:
                requested.add(reverse[name])
        result = subset_glyf(
            font.getTableData("glyf"),
            font["loca"],
            requested,
            include_notdef=self.options.notdef_glyph,
        )
        names = [glyph_order[gid] for gid in result.glyph_ids]
        new_cmap = {
            codepoint: names[result.old_to_new[reverse[name]]]
            for codepoint, name in cmap.items()
            if reverse[name] in result.old_to_new
        }
        font._load(_build_subset_font(font, result, names, new_cmap))
        self.glyphs_retained = set(names)
        return True


def _checksum(data):
    padded = data + b"\0" * ((-len(data)) & 3)
    return sum(struct.unpack(f">{len(padded) // 4}I", padded)) & 0xFFFFFFFF


def _build_cmap(cmap, name_to_gid):
    entries = sorted((cp, name_to_gid[name]) for cp, name in cmap.items())
    groups = []
    for cp, gid in entries:
        if groups and cp == groups[-1][1] + 1 and gid == groups[-1][2] + cp - groups[-1][0]:
            groups[-1] = (groups[-1][0], cp, groups[-1][2])
        else:
            groups.append((cp, cp, gid))
    subtable = bytearray(struct.pack(">HHIII", 12, 0, 16 + 12 * len(groups), 0, len(groups)))
    for group in groups:
        subtable.extend(struct.pack(">III", *group))
    return struct.pack(">HHHHI", 0, 1, 3, 10, 12) + subtable


def _build_post(original, names):
    header = bytearray(original[:32] if len(original) >= 32 else b"\0" * 32)
    struct.pack_into(">I", header, 0, 0x00020000)
    result = header + struct.pack(">H", len(names))
    standard = {name: index for index, name in enumerate(STANDARD_GLYPH_ORDER)}
    custom, indices = [], []
    for name in names:
        if name in standard:
            indices.append(standard[name])
        else:
            encoded = name.encode("latin-1")
            if len(encoded) > 255:
                raise SubsetError("PostScript glyph names must fit in 255 bytes")
            indices.append(258 + len(custom))
            custom.append(encoded)
    result.extend(struct.pack(f">{len(indices)}H", *indices))
    for name in custom:
        result.append(len(name))
        result.extend(name)
    return bytes(result)


def _build_subset_font(font, subset, names, cmap):
    tables = {}
    head = bytearray(font.getTableData("head"))
    struct.pack_into(">I", head, 8, 0)
    struct.pack_into(">h", head, 50, 1)
    tables["head"] = bytes(head)
    maxp = bytearray(font.getTableData("maxp"))
    struct.pack_into(">H", maxp, 4, len(names))
    tables["maxp"] = bytes(maxp)
    hhea = bytearray(font.getTableData("hhea"))
    struct.pack_into(">H", hhea, 34, len(names))
    tables["hhea"] = bytes(hhea)
    old_metrics = font["hmtx"].metrics
    tables["hmtx"] = b"".join(struct.pack(">Hh", *old_metrics[name]) for name in names)
    tables["glyf"] = subset.glyf
    tables["loca"] = struct.pack(f">{len(subset.loca)}I", *subset.loca)
    tables["cmap"] = _build_cmap(cmap, {name: gid for gid, name in enumerate(names)})
    tables["post"] = _build_post(font.getTableData("post") if "post" in font else b"", names)
    for tag in ("name", "OS/2", "cvt ", "fpgm", "prep", "gasp"):
        if tag in font:
            tables[tag] = font.getTableData(tag)
    tags = sorted(tables)
    count = len(tags)
    power = 2 ** int(math.log2(count))
    header = bytearray(font.sfntVersion)
    header.extend(struct.pack(">HHHH", count, power * 16, int(math.log2(power)), count * 16 - power * 16))
    directory = bytearray()
    body = bytearray()
    offset = 12 + count * 16
    head_offset = None
    for tag in tags:
        raw = tables[tag]
        directory.extend(struct.pack(">4sIII", tag.encode("latin-1"), _checksum(raw), offset, len(raw)))
        if tag == "head":
            head_offset = offset
        body.extend(raw)
        body.extend(b"\0" * ((-len(raw)) & 3))
        offset += (len(raw) + 3) & ~3
    result = header + directory + body
    adjustment = (0xB1B0AFBA - _checksum(bytes(result))) & 0xFFFFFFFF
    struct.pack_into(">I", result, head_offset + 8, adjustment)
    return bytes(result)


__all__ = ["GlyfSubset", "Options", "SubsetError", "Subsetter", "subset_glyf"]
