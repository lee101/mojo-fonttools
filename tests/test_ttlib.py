import io
import struct

import pytest
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont as ReferenceTTFont

from mojo_fonttools import TTFont
from mojo_fonttools.ttLib.sfnt import Glyph, TTLibError


def test_table_directory_and_metadata_parity(test_font_path):
    mojo = TTFont(test_font_path)
    reference = ReferenceTTFont(test_font_path)
    assert set(mojo.keys()) == set(reference.keys()) - {"GlyphOrder"}
    assert mojo["head"].unitsPerEm == reference["head"].unitsPerEm
    assert mojo["head"].indexToLocFormat == reference["head"].indexToLocFormat
    assert mojo["maxp"].numGlyphs == reference["maxp"].numGlyphs
    assert mojo["hhea"].ascent == reference["hhea"].ascent
    assert mojo.getGlyphOrder() == reference.getGlyphOrder()
    assert mojo.getBestCmap() == reference.getBestCmap()
    assert mojo["hmtx"].metrics == reference["hmtx"].metrics


@pytest.mark.parametrize("name", ["A", "acute", "curve"])
def test_simple_glyph_coordinates_parity(test_font_path, name):
    mojo = TTFont(test_font_path)
    reference = ReferenceTTFont(test_font_path)
    got = mojo["glyf"][name].getCoordinates(mojo["glyf"])
    expected = reference["glyf"][name].getCoordinates(reference["glyf"])
    assert list(got[0]) == list(expected[0])
    assert got[1] == list(expected[1])
    assert list(got[2]) == list(expected[2])


@pytest.mark.parametrize("name", ["A", "curve", "Aacute"])
def test_glyph_set_draw_parity(test_font_path, name):
    mojo = TTFont(test_font_path)
    reference = ReferenceTTFont(test_font_path)
    got_pen, expected_pen = RecordingPen(), RecordingPen()
    mojo.getGlyphSet()[name].draw(got_pen)
    reference.getGlyphSet()[name].draw(expected_pen)
    assert got_pen.value == expected_pen.value
    assert mojo.getGlyphSet()[name].width == reference.getGlyphSet()[name].width


def test_bytes_file_object_and_save_roundtrip(test_font_path):
    raw = test_font_path.read_bytes()
    font = TTFont(io.BytesIO(raw))
    output = io.BytesIO()
    font.save(output)
    assert output.getvalue() == raw
    assert TTFont(raw).getBestCmap() == font.getBestCmap()


def test_lookup_methods_and_context_manager(test_font_path):
    with TTFont(test_font_path) as font:
        gid = font.getGlyphID("A")
        assert font.getGlyphName(gid) == "A"
        assert font.getReverseGlyphMap()["A"] == gid
        assert font.getTableData("head")[:4] == b"\x00\x01\x00\x00"


def test_short_input_tables_report_ttlib_error(test_font_path):
    raw = bytearray(test_font_path.read_bytes())
    table_count = struct.unpack_from(">H", raw, 4)[0]
    for index in range(table_count):
        entry = 12 + index * 16
        if raw[entry : entry + 4] == b"head":
            struct.pack_into(">I", raw, entry + 12, 10)
            break
    with pytest.raises(TTLibError, match="truncated required"):
        TTFont(bytes(raw))


def test_loca_outside_glyf_is_rejected(test_font_path):
    raw = bytearray(test_font_path.read_bytes())
    font = TTFont(raw)
    loca_record = font.reader.tables["loca"]
    head_record = font.reader.tables["head"]
    glyf_length = font.reader.tables["glyf"].length
    loca_format = struct.unpack_from(">h", raw, head_record.offset + 50)[0]
    if loca_format == 0:
        struct.pack_into(">H", raw, loca_record.offset + loca_record.length - 2, glyf_length // 2 + 1)
    else:
        struct.pack_into(">I", raw, loca_record.offset + loca_record.length - 4, glyf_length + 1)
    with pytest.raises(TTLibError, match="outside glyf"):
        TTFont(bytes(raw))


def test_truncated_composite_reports_ttlib_error():
    data = struct.pack(">hhhhhHH", -1, 0, 0, 0, 0, 0x0003, 0)
    glyph = Glyph(None, 7, data)
    with pytest.raises(TTLibError, match="truncated composite"):
        glyph.getComponentInfo()
