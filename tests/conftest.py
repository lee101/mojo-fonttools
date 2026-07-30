import os
import sys

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)


@pytest.fixture
def test_font_path(tmp_path):
    glyph_order = [".notdef", "space", "A", "acute", "Aacute", "curve"]
    glyphs = {}

    pen = TTGlyphPen(None)
    glyphs[".notdef"] = pen.glyph()
    glyphs["space"] = TTGlyphPen(None).glyph()

    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((300, 700))
    pen.lineTo((550, 0))
    pen.lineTo((450, 0))
    pen.lineTo((380, 210))
    pen.lineTo((220, 210))
    pen.lineTo((150, 0))
    pen.closePath()
    glyphs["A"] = pen.glyph()

    pen = TTGlyphPen(None)
    pen.moveTo((200, 730))
    pen.lineTo((340, 870))
    pen.lineTo((430, 870))
    pen.lineTo((270, 730))
    pen.closePath()
    glyphs["acute"] = pen.glyph()

    pen = TTGlyphPen(glyphs)
    pen.addComponent("A", (0.75, 0, 0, 1.25, 10, 20))
    pen.addComponent("acute", (1, 0, 0, 1, 0, 0))
    glyphs["Aacute"] = pen.glyph()

    pen = TTGlyphPen(None)
    pen.moveTo((0, 0))
    pen.qCurveTo((250, 600), (500, 0))
    pen.qCurveTo((250, -200), (0, 0))
    pen.closePath()
    glyphs["curve"] = pen.glyph()

    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(glyph_order)
    builder.setupCharacterMap({0x20: "space", 0x41: "A", 0xB4: "acute", 0xC1: "Aacute", 0x2605: "curve"})
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(
        {
            ".notdef": (600, 0),
            "space": (300, 0),
            "A": (600, 50),
            "acute": (500, 200),
            "Aacute": (600, 50),
            "curve": (520, 0),
        }
    )
    builder.setupHorizontalHeader(ascent=900, descent=-250)
    builder.setupNameTable(
        {
            "familyName": "Mojo Test",
            "styleName": "Regular",
            "uniqueFontIdentifier": "Mojo Test Regular",
            "fullName": "Mojo Test Regular",
            "psName": "MojoTest-Regular",
        }
    )
    builder.setupOS2(
        sTypoAscender=900,
        sTypoDescender=-250,
        usWinAscent=900,
        usWinDescent=250,
    )
    builder.setupPost()
    path = tmp_path / "test.ttf"
    builder.save(path)
    return path
