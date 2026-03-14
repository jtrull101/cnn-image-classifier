"""Tiny PNG image factory for upload tests (no PIL dependency)."""

import struct
import zlib
from pathlib import Path


def create_png_1x1() -> bytes:
    """Return a valid minimal 1×1 pixel RGB PNG image as bytes."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        header = struct.pack(">I", len(data)) + tag + data
        return header + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    # IHDR: width=1, height=1, bit_depth=8, color_type=2 (RGB), compression=0, filter=0, interlace=0
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    # IDAT: filter byte (0) + R=255, G=0, B=0 (one red pixel)
    png += _chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    png += _chunk(b"IEND", b"")
    return png


def write_test_png(dest: Path) -> Path:
    """Write a 1×1 PNG to *dest* and return the path."""
    dest.write_bytes(create_png_1x1())
    return dest
