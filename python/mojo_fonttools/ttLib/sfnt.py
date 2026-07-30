from __future__ import annotations

import io
import os
import struct
from types import SimpleNamespace

import numpy as np

from .._lib import addr, lib

STANDARD_GLYPH_ORDER = [
    ".notdef", ".null", "nonmarkingreturn", "space", "exclam", "quotedbl",
    "numbersign", "dollar", "percent", "ampersand", "quotesingle", "parenleft",
    "parenright", "asterisk", "plus", "comma", "hyphen", "period", "slash",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "colon", "semicolon", "less", "equal", "greater", "question", "at",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "bracketleft", "backslash", "bracketright", "asciicircum", "underscore",
    "grave", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "braceleft", "bar", "braceright", "asciitilde", "Adieresis", "Aring",
    "Ccedilla", "Eacute", "Ntilde", "Odieresis", "Udieresis", "aacute",
    "agrave", "acircumflex", "adieresis", "atilde", "aring", "ccedilla",
    "eacute", "egrave", "ecircumflex", "edieresis", "iacute", "igrave",
    "icircumflex", "idieresis", "ntilde", "oacute", "ograve", "ocircumflex",
    "odieresis", "otilde", "uacute", "ugrave", "ucircumflex", "udieresis",
    "dagger", "degree", "cent", "sterling", "section", "bullet", "paragraph",
    "germandbls", "registered", "copyright", "trademark", "acute", "dieresis",
    "notequal", "AE", "Oslash", "infinity", "plusminus", "lessequal",
    "greaterequal", "yen", "mu", "partialdiff", "summation", "product", "pi",
    "integral", "ordfeminine", "ordmasculine", "Omega", "ae", "oslash",
    "questiondown", "exclamdown", "logicalnot", "radical", "florin",
    "approxequal", "Delta", "guillemotleft", "guillemotright", "ellipsis",
    "nonbreakingspace", "Agrave", "Atilde", "Otilde", "OE", "oe", "endash",
    "emdash", "quotedblleft", "quotedblright", "quoteleft", "quoteright",
    "divide", "lozenge", "ydieresis", "Ydieresis", "fraction", "currency",
    "guilsinglleft", "guilsinglright", "fi", "fl", "daggerdbl",
    "periodcentered", "quotesinglbase", "quotedblbase", "perthousand",
    "Acircumflex", "Ecircumflex", "Aacute", "Edieresis", "Egrave", "Iacute",
    "Icircumflex", "Idieresis", "Igrave", "Oacute", "Ocircumflex", "apple",
    "Ograve", "Uacute", "Ucircumflex", "Ugrave", "dotlessi", "circumflex",
    "tilde", "macron", "breve", "dotaccent", "ring", "cedilla",
    "hungarumlaut", "ogonek", "caron", "Lslash", "lslash", "Scaron",
    "scaron", "Zcaron", "zcaron", "brokenbar", "Eth", "eth", "Yacute",
    "yacute", "Thorn", "thorn", "minus", "multiply", "onesuperior",
    "twosuperior", "threesuperior", "onehalf", "onequarter", "threequarters",
    "franc", "Gbreve", "gbreve", "Idotaccent", "Scedilla", "scedilla",
    "Cacute", "cacute", "Ccaron", "ccaron", "dcroat",
]


class TTLibError(Exception):
    pass


def _u16(data, offset):
    return struct.unpack_from(">H", data, offset)[0]


def _i16(data, offset):
    return struct.unpack_from(">h", data, offset)[0]


def _u32(data, offset):
    return struct.unpack_from(">I", data, offset)[0]


def _f2dot14(data, offset):
    return _i16(data, offset) / 16384.0


class GlyphCoordinates(list):
    def __init__(self, values=()):
        super().__init__(values)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return GlyphCoordinates(value) if isinstance(item, slice) else value


class Glyph:
    def __init__(self, table, gid, data):
        self._table = table
        self.gid = gid
        self.data = data
        if len(data) >= 10:
            (
                self.numberOfContours,
                self.xMin,
                self.yMin,
                self.xMax,
                self.yMax,
            ) = struct.unpack_from(">hhhhh", data)
        else:
            self.numberOfContours = 0
            self.xMin = self.yMin = self.xMax = self.yMax = 0

    def isComposite(self):
        return self.numberOfContours < 0

    def getCoordinates(self, glyfTable=None):
        if self.isComposite():
            raise TTLibError("getCoordinates for composites requires decomposition; use draw")
        if self.numberOfContours == 0 or not self.data:
            return GlyphCoordinates(), [], bytearray()
        ends = list(struct.unpack_from(f">{self.numberOfContours}H", self.data, 10))
        count = ends[-1] + 1
        source = np.frombuffer(self.data, dtype=np.uint8)
        xs, ys = np.empty(count), np.empty(count)
        flags = np.empty(count, dtype=np.uint8)
        consumed = lib().mft_decode_simple_glyph(
            addr(source, np.uint8), len(source), self.numberOfContours, count,
            addr(xs, np.float64, writable=True),
            addr(ys, np.float64, writable=True),
            addr(flags, np.uint8, writable=True),
        )
        if consumed < 0:
            raise TTLibError(f"malformed simple glyph {self.gid}")
        coords = GlyphCoordinates(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))
        return coords, ends, bytearray((int(flag) & 0xC1) for flag in flags)

    def getComponentInfo(self):
        if not self.isComposite():
            return []
        data, pos, result = self.data, 10, []
        more = True
        while more:
            if pos + 4 > len(data):
                raise TTLibError(f"truncated composite glyph {self.gid}")
            flags, component_gid = struct.unpack_from(">HH", data, pos)
            pos += 4
            words = bool(flags & 0x0001)
            xy = bool(flags & 0x0002)
            if words:
                if pos + 4 > len(data):
                    raise TTLibError(f"truncated composite glyph {self.gid}")
                arg1, arg2 = struct.unpack_from(">hh" if xy else ">HH", data, pos)
                pos += 4
            else:
                if pos + 2 > len(data):
                    raise TTLibError(f"truncated composite glyph {self.gid}")
                arg1, arg2 = struct.unpack_from(">bb" if xy else ">BB", data, pos)
                pos += 2
            if not xy:
                raise TTLibError("point-attached composite components are not covered")
            xx = yy = 1.0
            xy_scale = yx_scale = 0.0
            if flags & 0x0008:
                if pos + 2 > len(data):
                    raise TTLibError(f"truncated composite glyph {self.gid}")
                xx = yy = _f2dot14(data, pos)
                pos += 2
            elif flags & 0x0040:
                if pos + 4 > len(data):
                    raise TTLibError(f"truncated composite glyph {self.gid}")
                xx, yy = _f2dot14(data, pos), _f2dot14(data, pos + 2)
                pos += 4
            elif flags & 0x0080:
                if pos + 8 > len(data):
                    raise TTLibError(f"truncated composite glyph {self.gid}")
                xx, xy_scale, yx_scale, yy = (
                    _f2dot14(data, pos + i * 2) for i in range(4)
                )
                pos += 8
            result.append(
                (
                    component_gid,
                    (xx, xy_scale, yx_scale, yy, float(arg1), float(arg2)),
                    flags,
                )
            )
            more = bool(flags & 0x0020)
        return result

    def draw(self, pen, glyfTable=None, offset=0):
        table = glyfTable or self._table
        if self.isComposite():
            for gid, transform, _ in self.getComponentInfo():
                pen.addComponent(table.glyphOrder[gid], transform)
            return
        coords, ends, flags = self.getCoordinates(table)
        start = 0
        for end in ends:
            points = coords[start : end + 1]
            on_curve = [(flags[i] & 1) != 0 for i in range(start, end + 1)]
            self._draw_contour(pen, points, on_curve)
            start = end + 1

    @staticmethod
    def _draw_contour(pen, points, on_curve):
        if not points:
            return
        if on_curve[0]:
            start = points[0]
            sequence = list(zip(points[1:], on_curve[1:]))
        elif on_curve[-1]:
            start = points[-1]
            sequence = list(zip(points[:-1], on_curve[:-1]))
        else:
            start = (
                (points[-1][0] + points[0][0]) / 2,
                (points[-1][1] + points[0][1]) / 2,
            )
            sequence = list(zip(points, on_curve))
        pen.moveTo(start)
        index = 0
        while index < len(sequence):
            point, is_on = sequence[index]
            if is_on:
                pen.lineTo(point)
                index += 1
                continue
            if index + 1 < len(sequence):
                next_point, next_on = sequence[index + 1]
            else:
                next_point, next_on = start, True
            if next_on:
                pen.qCurveTo(point, next_point)
                index += 2
            else:
                implied = (
                    (point[0] + next_point[0]) / 2,
                    (point[1] + next_point[1]) / 2,
                )
                pen.qCurveTo(point, implied)
                index += 1
        pen.closePath()


class GlyfTable:
    def __init__(self, data, loca, glyph_order):
        self.data = data
        self.loca = loca
        self.glyphOrder = glyph_order
        self._name_to_gid = {name: gid for gid, name in enumerate(glyph_order)}
        self._cache = {}

    def __getitem__(self, name):
        gid = name if isinstance(name, int) else self._name_to_gid[name]
        if gid not in self._cache:
            self._cache[gid] = Glyph(self, gid, self.data[self.loca[gid] : self.loca[gid + 1]])
        return self._cache[gid]

    def __contains__(self, name):
        return name in self._name_to_gid


class _GlyphSet:
    def __init__(self, font):
        self.font = font

    def __getitem__(self, name):
        glyph = self.font["glyf"][name]
        width, lsb = self.font["hmtx"].metrics[name]
        return _GlyphView(glyph, self.font["glyf"], width, lsb)

    def __contains__(self, name):
        return name in self.font["glyf"]

    def keys(self):
        return self.font.getGlyphOrder()


class _GlyphView:
    def __init__(self, glyph, table, width, lsb):
        self._glyph, self._table = glyph, table
        self.width, self.lsb = width, lsb

    def draw(self, pen):
        self._glyph.draw(pen, self._table)


class TTFont:
    def __init__(
        self, file=None, res_name_or_index=None, sfntVersion="\x00\x01\x00\x00",
        flavor=None, checkChecksums=0, verbose=None, recalcBBoxes=True,
        allowVID=False, ignoreDecompileErrors=False, recalcTimestamp=True,
        fontNumber=-1, lazy=None, quiet=None, ignoreDecompileErrorsDefault=False,
        cfg=None,
    ):
        if file is None:
            raise TTLibError("creating empty fonts is not covered")
        if isinstance(file, (str, os.PathLike)):
            with open(file, "rb") as stream:
                data = stream.read()
        elif isinstance(file, (bytes, bytearray, memoryview)):
            data = bytes(file)
        else:
            data = file.read()
        self._load(data)

    def _load(self, data):
        self._data = bytes(data)
        if len(data) < 12 or data[:4] not in (b"\x00\x01\x00\x00", b"true"):
            raise TTLibError("only standalone TrueType-flavored SFNT fonts are covered")
        count = _u16(data, 4)
        if 12 + count * 16 > len(data):
            raise TTLibError("truncated SFNT directory")
        self.sfntVersion = data[:4]
        self.reader = SimpleNamespace(tables={})
        self._tables = {}
        for index in range(count):
            tag, checksum, offset, length = struct.unpack_from(">4sIII", data, 12 + 16 * index)
            if offset + length > len(data):
                raise TTLibError(f"table {tag!r} is outside the file")
            name = tag.decode("latin-1")
            self.reader.tables[name] = SimpleNamespace(
                checkSum=checksum, offset=offset, length=length
            )
            self._tables[name] = bytes(data[offset : offset + length])
        for required in ("head", "maxp", "loca", "glyf", "hhea", "hmtx"):
            if required not in self._tables:
                raise TTLibError(f"required TrueType table {required!r} is missing")
        head, maxp, hhea = self._tables["head"], self._tables["maxp"], self._tables["hhea"]
        if len(head) < 54 or len(maxp) < 6 or len(hhea) < 36:
            raise TTLibError("truncated required TrueType metadata table")
        num_glyphs = _u16(maxp, 4)
        loca_format = _i16(head, 50)
        loca_data = self._tables["loca"]
        if loca_format == 0:
            loca_size = 2 * (num_glyphs + 1)
            if len(loca_data) < loca_size:
                raise TTLibError("truncated short loca table")
            loca = [value * 2 for value in struct.unpack(f">{num_glyphs + 1}H", loca_data[:loca_size])]
        elif loca_format == 1:
            loca_size = 4 * (num_glyphs + 1)
            if len(loca_data) < loca_size:
                raise TTLibError("truncated long loca table")
            loca = list(struct.unpack(f">{num_glyphs + 1}I", loca_data[:loca_size]))
        else:
            raise TTLibError(f"invalid indexToLocFormat {loca_format}")
        if loca[0] != 0 or any(a > b for a, b in zip(loca, loca[1:])):
            raise TTLibError("loca offsets must be sorted and start at zero")
        if loca[-1] > len(self._tables["glyf"]):
            raise TTLibError("loca table points outside glyf data")
        self._glyph_order = self._parse_post_names(num_glyphs)
        self._cmap = self._parse_cmap()
        self._tables_decoded = {
            "head": SimpleNamespace(
                unitsPerEm=_u16(head, 18), xMin=_i16(head, 36), yMin=_i16(head, 38),
                xMax=_i16(head, 40), yMax=_i16(head, 42), indexToLocFormat=loca_format,
            ),
            "maxp": SimpleNamespace(numGlyphs=num_glyphs),
            "hhea": SimpleNamespace(
                ascent=_i16(hhea, 4), descent=_i16(hhea, 6),
                numberOfHMetrics=_u16(hhea, 34),
            ),
        }
        self._loca = loca
        self._tables_decoded["loca"] = loca
        self._tables_decoded["glyf"] = GlyfTable(self._tables["glyf"], loca, self._glyph_order)
        self._tables_decoded["hmtx"] = SimpleNamespace(metrics=self._parse_hmtx())

    def _parse_post_names(self, num_glyphs):
        post = self._tables.get("post")
        fallback = [".notdef"] + [f"glyph{i:05d}" for i in range(1, num_glyphs)]
        if not post or len(post) < 34 or _u32(post, 0) != 0x00020000:
            return fallback
        count = _u16(post, 32)
        if count != num_glyphs or len(post) < 34 + 2 * count:
            return fallback
        indices = struct.unpack_from(f">{count}H", post, 34)
        pos, custom = 34 + 2 * count, []
        needed = max(indices, default=257) - 257
        for _ in range(max(0, needed)):
            if pos >= len(post):
                return fallback
            size = post[pos]
            pos += 1
            if pos + size > len(post):
                return fallback
            custom.append(post[pos : pos + size].decode("latin-1"))
            pos += size
        names = []
        for gid, index in enumerate(indices):
            if index < len(STANDARD_GLYPH_ORDER):
                names.append(STANDARD_GLYPH_ORDER[index])
            elif index - 258 < len(custom):
                names.append(custom[index - 258])
            else:
                names.append(f"glyph{gid:05d}")
        return names

    def _parse_hmtx(self):
        data = self._tables["hmtx"]
        metric_count = self._tables_decoded["hhea"].numberOfHMetrics
        num_glyphs = len(self._glyph_order)
        if not 1 <= metric_count <= num_glyphs:
            raise TTLibError("invalid numberOfHMetrics")
        required = metric_count * 4 + (num_glyphs - metric_count) * 2
        if len(data) < required:
            raise TTLibError("truncated hmtx table")
        metrics, advance = {}, 0
        for gid, name in enumerate(self._glyph_order):
            if gid < metric_count:
                advance, lsb = struct.unpack_from(">Hh", data, gid * 4)
            else:
                lsb = _i16(data, metric_count * 4 + (gid - metric_count) * 2)
            metrics[name] = (advance, lsb)
        return metrics

    def _parse_cmap(self):
        data = self._tables.get("cmap")
        if not data or len(data) < 4:
            return {}
        record_count = _u16(data, 2)
        if 4 + record_count * 8 > len(data):
            raise TTLibError("truncated cmap encoding records")
        records = []
        for i in range(record_count):
            platform, encoding, offset = struct.unpack_from(">HHI", data, 4 + i * 8)
            if offset + 2 <= len(data):
                priority = (
                    4 if (platform, encoding) == (3, 10) else
                    3 if platform == 0 else
                    2 if (platform, encoding) == (3, 1) else 0
                )
                records.append((priority, offset))
        for _, offset in sorted(records, reverse=True):
            fmt = _u16(data, offset)
            if fmt == 12:
                if offset + 16 > len(data):
                    raise TTLibError("truncated cmap format 12 header")
                result = {}
                length = _u32(data, offset + 4)
                groups = _u32(data, offset + 12)
                if length < 16 or offset + length > len(data) or 16 + groups * 12 > length:
                    raise TTLibError("truncated cmap format 12 groups")
                for i in range(groups):
                    start, end, gid = struct.unpack_from(">III", data, offset + 16 + i * 12)
                    if start > end:
                        raise TTLibError("invalid cmap format 12 group")
                    for cp in range(start, end + 1):
                        if gid + cp - start < len(self._glyph_order):
                            result[cp] = self._glyph_order[gid + cp - start]
                return result
            if fmt == 4:
                return self._parse_cmap4(data, offset)
        return {}

    def _parse_cmap4(self, data, offset):
        if offset + 14 > len(data):
            raise TTLibError("truncated cmap format 4 header")
        length = _u16(data, offset + 2)
        table_end = offset + length
        if length < 16 or table_end > len(data):
            raise TTLibError("truncated cmap format 4 table")
        seg_count = _u16(data, offset + 6) // 2
        if seg_count == 0 or offset + 16 + seg_count * 8 > table_end:
            raise TTLibError("invalid cmap format 4 segments")
        end_pos = offset + 14
        start_pos = end_pos + 2 * seg_count + 2
        delta_pos = start_pos + 2 * seg_count
        range_pos = delta_pos + 2 * seg_count
        result = {}
        for i in range(seg_count):
            end, start = _u16(data, end_pos + 2 * i), _u16(data, start_pos + 2 * i)
            delta, range_offset = _i16(data, delta_pos + 2 * i), _u16(data, range_pos + 2 * i)
            for cp in range(start, min(end, 0xFFFE) + 1):
                if range_offset:
                    glyph_pos = range_pos + 2 * i + range_offset + 2 * (cp - start)
                    gid = _u16(data, glyph_pos) if glyph_pos + 2 <= table_end else 0
                    gid = (gid + delta) & 0xFFFF if gid else 0
                else:
                    gid = (cp + delta) & 0xFFFF
                if 0 < gid < len(self._glyph_order):
                    result[cp] = self._glyph_order[gid]
        return result

    def keys(self):
        return list(self._tables)

    def __contains__(self, tag):
        return tag in self._tables

    def __getitem__(self, tag):
        if tag in self._tables_decoded:
            return self._tables_decoded[tag]
        raise KeyError(f"table {tag!r} is present but not decoded by mojo-fonttools")

    def getTableData(self, tag):
        return self._tables[tag]

    def getGlyphOrder(self):
        return list(self._glyph_order)

    def setGlyphOrder(self, glyphOrder):
        raise TTLibError("setGlyphOrder is only supported internally by Subsetter")

    def getGlyphID(self, glyphName):
        return self._glyph_order.index(glyphName)

    def getGlyphName(self, glyphID):
        return self._glyph_order[glyphID]

    def getReverseGlyphMap(self, rebuild=False):
        return {name: gid for gid, name in enumerate(self._glyph_order)}

    def getBestCmap(self, cmapPreferences=None):
        return dict(self._cmap)

    def getGlyphSet(self, preferCFF=True, location=None, normalized=False, recalcBounds=True):
        return _GlyphSet(self)

    def save(self, file, reorderTables=True):
        if isinstance(file, (str, os.PathLike)):
            with open(file, "wb") as stream:
                stream.write(self._data)
        else:
            file.write(self._data)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
