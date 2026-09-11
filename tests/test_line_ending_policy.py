"""Checksum-locked shaders must hold the same bytes as their blobs.

Committed reports pin shader sources by hashing what is on disk. With
core.autocrlf enabled and no attributes entry, a checkout on Windows writes
CRLF while the blob keeps LF, so the hash stops matching and a fresh clone
fails the suite immediately. This pins the narrow policy that prevents it.

The scope is deliberate. Other tracked text in this repository already holds
CRLF on disk and is not checksum-locked, so normalising it would be a
repository-wide rewrite rather than a fix for a demonstrated failure.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ATTRIBUTES = REPOSITORY_ROOT / ".gitattributes"
LOCKED_REPORT = REPOSITORY_ROOT / (
    "docs/validation/road_markings_v2/metric_road_marking_report.json"
)


def _locked_sources() -> list[tuple[str, str]]:
    report = json.loads(LOCKED_REPORT.read_text(encoding="utf-8"))
    return [
        (source["asset"], source["sha256"])
        for source in report["implementation_sources"].values()
    ]


def test_the_shader_types_are_pinned_to_lf() -> None:
    assert ATTRIBUTES.is_file(), "a .gitattributes is required with autocrlf on"
    text = ATTRIBUTES.read_text(encoding="utf-8")
    for suffix in (".frag", ".vert", ".glsl"):
        assert f"*{suffix} text eol=lf" in text, suffix


def test_locked_sources_match_both_their_hash_and_their_blob() -> None:
    sources = _locked_sources()
    assert sources, "the report locks no implementation source"

    for relative, expected in sources:
        path = REPOSITORY_ROOT / relative
        assert path.is_file(), relative
        on_disk = path.read_bytes()
        assert sha256(on_disk).hexdigest() == expected, relative
        result = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
        )
        if result.returncode != 0:
            continue
        # The failure mode this exists for: identical text, different endings.
        assert on_disk == result.stdout, (
            f"{relative} differs from its committed blob; check line endings"
        )


def test_no_locked_shader_holds_crlf() -> None:
    for relative, _ in _locked_sources():
        path = REPOSITORY_ROOT / relative
        if path.suffix not in (".frag", ".vert", ".glsl"):
            continue
        assert b"\r\n" not in path.read_bytes(), relative
