import io
import struct

import pytest
from fontTools.pens.recordingPen import RecordingPen
from fontTools.subset import Options as ReferenceOptions
from fontTools.subset import Subsetter as ReferenceSubsetter
from fontTools.ttLib import TTFont as ReferenceTTFont

from mojo_fonttools import TTFont
from mojo_fonttools.subset import SubsetError, Subsetter, subset_glyf


def test_raw_glyf_subset_closes_composites_and_rewrites_ids(test_font_path):
    font = TTFont(test_font_path)
    composite_gid = font.getGlyphID("Aacute")
    result = subset_glyf(font.getTableData("glyf"), font["loca"], [composite_gid])
    kept_names = [font.getGlyphName(gid) for gid in result.glyph_ids]
    assert kept_names == [".notdef", "A", "acute", "Aacute"]
    new_composite = result.glyf[result.loca[-2] : result.loca[-1]]
    first_component_gid = struct.unpack_from(">H", new_composite, 12)[0]
    assert first_component_gid == kept_names.index("A")


def test_subsetter_produces_upstream_readable_font(test_font_path):
    font = TTFont(test_font_path)
    subsetter = Subsetter()
    subsetter.populate(text="Á")
    assert subsetter.subset(font)
    output = io.BytesIO()
    font.save(output)
    output.seek(0)
    parsed = ReferenceTTFont(output)
    assert parsed.getGlyphOrder() == [".notdef", "A", "acute", "Aacute"]
    assert parsed.getBestCmap() == {0x41: "A", 0xB4: "acute", 0xC1: "Aacute"}
    pen = RecordingPen()
    parsed.getGlyphSet()["Aacute"].draw(pen)
    assert [command for command, _ in pen.value] == ["addComponent", "addComponent"]


def test_subset_behavior_matches_upstream_for_selected_text(test_font_path):
    ours = TTFont(test_font_path)
    our_subsetter = Subsetter()
    our_subsetter.populate(text="A★")
    our_subsetter.subset(ours)

    reference = ReferenceTTFont(test_font_path)
    options = ReferenceOptions()
    options.layout_features = []
    ref_subsetter = ReferenceSubsetter(options)
    ref_subsetter.populate(text="A★")
    ref_subsetter.subset(reference)

    assert ours.getGlyphOrder() == reference.getGlyphOrder()
    assert ours.getBestCmap() == reference.getBestCmap()
    for name in ours.getGlyphOrder():
        got, expected = RecordingPen(), RecordingPen()
        ours.getGlyphSet()[name].draw(got)
        reference.getGlyphSet()[name].draw(expected)
        assert got.value == expected.value


@pytest.mark.parametrize(
    "populate_kwargs, expected",
    [
        ({"glyphs": ["A"]}, {".notdef", "A"}),
        ({"gids": [2]}, {".notdef", "A"}),
        ({"unicodes": [0x41]}, {".notdef", "A"}),
    ],
)
def test_populate_variants(test_font_path, populate_kwargs, expected):
    font = TTFont(test_font_path)
    subsetter = Subsetter()
    subsetter.populate(**populate_kwargs)
    subsetter.subset(font)
    assert set(font.getGlyphOrder()) == expected


def test_raw_glyf_rejects_loca_outside_data():
    with pytest.raises(SubsetError, match="outside glyf"):
        subset_glyf(b"\0" * 10, [0, 12], [0])


def test_notdef_option_controls_retention(test_font_path):
    font = TTFont(test_font_path)
    subsetter = Subsetter()
    subsetter.options.notdef_glyph = False
    subsetter.populate(glyphs=["A"])
    subsetter.subset(font)
    assert font.getGlyphOrder() == ["A"]


def test_identifiers_do_not_silently_narrow_floats():
    with pytest.raises(SubsetError, match="must be integers"):
        subset_glyf(b"", [0, 0], [0.5])
    with pytest.raises(SubsetError, match="must be integers"):
        Subsetter().populate(unicodes=[65.5])
